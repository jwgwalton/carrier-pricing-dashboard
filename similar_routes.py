import polars as pl
import math
import requests

postcode_cache = {}

import requests

postcode_cache: dict[str, tuple[float, float]] = {}

def get_lat_lon_from_api(postcode: str) -> tuple[float, float]:
    """
    Retrieve the latitude and longitude for a given UK postcode using the postcodes.io API.

    Algorithm:
    1. Clean the postcode by removing spaces and converting to uppercase.
    2. If the postcode is cached, return the cached result.
    3. If the cleaned postcode length is greater than 4:
       - First attempt a full postcode lookup via the `/postcodes/` endpoint.
       - If that fails, fallback to the first 4 characters via the `/outcodes/` endpoint.
    4. If the postcode is 4 characters or fewer, or if the above attempts fail:
       - Fallback to the first 3 characters via the `/outcodes/` endpoint.
    5. If all attempts fail, raise a ValueError.

    Parameters:
        postcode (str): The UK postcode to look up.

    Returns:
        tuple[float, float]: A tuple of (latitude, longitude) if found.

    Raises:
        ValueError: If no location data could be retrieved after all fallback attempts.
    """
    postcode = postcode.replace(" ", "").upper()

    if postcode in postcode_cache:
        return postcode_cache[postcode]

    def fetch_lat_lon(url: str) -> tuple[float, float] | None:
        resp = requests.get(url)
        if resp.status_code == 200:
            result = resp.json().get("result")
            if result:
                return (result["latitude"], result["longitude"])
        return None

    lat_lon = None

    if len(postcode) > 4:
        # Try full postcode first
        url = f"https://api.postcodes.io/postcodes/{postcode}"
        lat_lon = fetch_lat_lon(url)

        # Fallback to first 4 characters
        if lat_lon is None and len(postcode) >= 4:
            outcode_4 = postcode[:4]
            url_4 = f"https://api.postcodes.io/outcodes/{outcode_4}"
            lat_lon = fetch_lat_lon(url_4)

    # Final fallback to first 3 characters
    if lat_lon is None and len(postcode) >= 3:
        outcode_3 = postcode[:3]
        url_3 = f"https://api.postcodes.io/outcodes/{outcode_3}"
        lat_lon = fetch_lat_lon(url_3)

    if lat_lon is not None:
        postcode_cache[postcode] = lat_lon
        return lat_lon
    else:
        raise ValueError(f"Postcode '{postcode}' not found after all fallback attempts.")


def haversine_expr(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km

    lat1_rad = pl.col(lat1).radians()
    lon1_rad = pl.col(lon1).radians()

    # If lat2 and lon2 are scalars (float), convert to literals in radians
    lat2_rad = pl.lit(math.radians(lat2))
    lon2_rad = pl.lit(math.radians(lon2))

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        (dlat / 2).map_elements(math.sin, return_dtype=pl.Float64) ** 2
        + lat1_rad.map_elements(math.cos, return_dtype=pl.Float64)
        * pl.lit(math.cos(math.radians(lat2)))
        * (dlon / 2).map_elements(math.sin, return_dtype=pl.Float64) ** 2
    )

    c = a.map_elements(lambda x: 2 * math.asin(math.sqrt(x)), return_dtype=pl.Float64)
    return c * R

from typing import Optional

def find_similar_routes_by_postcode(
    df: pl.DataFrame,
    origin_postcode: str,
    dest_postcode: str,
    vehicle_type: Optional[str] = None
):
    origin_postcode_clean = origin_postcode.replace(" ", "").upper()
    dest_postcode_clean = dest_postcode.replace(" ", "").upper()

    origin_prefix_3 = origin_postcode_clean[:3]
    dest_prefix_3 = dest_postcode_clean[:3]
    origin_prefix_2 = origin_postcode_clean[:2]
    dest_prefix_2 = dest_postcode_clean[:2]

    target_origin_lat, target_origin_lon = get_lat_lon_from_api(origin_postcode_clean)
    target_dest_lat, target_dest_lon = get_lat_lon_from_api(dest_postcode_clean)

    df = df.with_columns([
        haversine_expr("origin_lat", "origin_lon", target_origin_lat, target_origin_lon).alias("forward_origin_dist"),
        haversine_expr("dest_lat", "dest_lon", target_dest_lat, target_dest_lon).alias("forward_dest_dist"),
        haversine_expr("origin_lat", "origin_lon", target_dest_lat, target_dest_lon).alias("reverse_origin_dist"),
        haversine_expr("dest_lat", "dest_lon", target_origin_lat, target_origin_lon).alias("reverse_dest_dist"),
        pl.col("origin_postcode").str.replace_all(" ", "").str.to_uppercase().alias("origin_postcode_clean"),
        pl.col("destination_postcode").str.replace_all(" ", "").str.to_uppercase().alias("destination_postcode_clean"),
    ])

    df = df.with_columns([
        pl.col("origin_postcode_clean").str.slice(0, 3).alias("origin_prefix_3"),
        pl.col("destination_postcode_clean").str.slice(0, 3).alias("dest_prefix_3"),
        pl.col("origin_postcode_clean").str.slice(0, 2).alias("origin_prefix_2"),
        pl.col("destination_postcode_clean").str.slice(0, 2).alias("dest_prefix_2"),
    ])

    required_cols = [
        "origin_postcode", "destination_postcode", "carrier_price",
        "vehicle_type", "pickup_date", "contract_type", "shipper_price", "carrier_name", "shipper_id"
    ]

    carrier_price_filter = (pl.col("carrier_price").is_not_null()) & (pl.col("carrier_price") != 0)

    # Conditionally apply vehicle_type filter
    if vehicle_type is not None:
        vehicle_filter = pl.col("vehicle_type") == vehicle_type
    else:
        vehicle_filter = pl.lit(True)  # No-op filter

    combined_filter = carrier_price_filter & vehicle_filter

    matches_3_letter = df.filter(
        combined_filter & (
            ((pl.col("origin_prefix_3") == origin_prefix_3) & (pl.col("dest_prefix_3") == dest_prefix_3)) |
            ((pl.col("origin_prefix_3") == dest_prefix_3) & (pl.col("dest_prefix_3") == origin_prefix_3))
        )
    ).select(required_cols)

    matches_2_letter = df.filter(
        combined_filter & (
            ((pl.col("origin_prefix_2") == origin_prefix_2) & (pl.col("dest_prefix_2") == dest_prefix_2)) |
            ((pl.col("origin_prefix_2") == dest_prefix_2) & (pl.col("dest_prefix_2") == origin_prefix_2))
        )
    ).select(required_cols)

    within_10km = df.filter(
        combined_filter & (
            ((pl.col("forward_origin_dist") <= 10.0) & (pl.col("forward_dest_dist") <= 10.0)) |
            ((pl.col("reverse_origin_dist") <= 10.0) & (pl.col("reverse_dest_dist") <= 10.0))
        )
    ).select(required_cols)

    within_20km = df.filter(
        combined_filter & (
            ((pl.col("forward_origin_dist") <= 20.0) & (pl.col("forward_dest_dist") <= 20.0)) |
            ((pl.col("reverse_origin_dist") <= 20.0) & (pl.col("reverse_dest_dist") <= 20.0))
        )
    ).select(required_cols)

    return matches_3_letter, within_10km, within_20km, matches_2_letter

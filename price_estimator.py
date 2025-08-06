import polars as pl
from typing import Optional, Dict, Any
from datetime import date
import math


def create_route_key(
    df: pl.DataFrame,
    origin_col: str = "origin_postcode",
    dest_col: str = "destination_postcode",
    output_col: str = "route_key"
) -> pl.DataFrame:
    """
    Create a non-directional route key column from two postcode columns.

    Parameters:
    - df: Polars DataFrame
    - origin_col: Name of the origin postcode column
    - dest_col: Name of the destination postcode column
    - output_col: Name of the new route key column
    """
    return df.with_columns([
        pl.when(pl.col(origin_col) < pl.col(dest_col))
        .then(pl.concat_str([pl.col(origin_col), pl.col(dest_col)], separator=" - "))
        .otherwise(pl.concat_str([pl.col(dest_col), pl.col(origin_col)], separator=" - "))
        .alias(output_col)
    ])

def estimate_price_from_df(
    df: pl.DataFrame,
    date_column="pickup_date",
    use_time_weighting: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Compute volume-weighted and optionally time-weighted cost statistics.
    Time weighting is based on how many years ago the job occurred.
    """
    if df.is_empty():
        return None

    df = create_route_key(df)

    filtered_df = df.filter(
        (pl.col("carrier_price").is_not_null()) &
        (pl.col("carrier_price") != 0) &
        (pl.col(date_column).is_not_null())
    ).with_columns([
        pl.col("carrier_price").cast(pl.Float64)
    ])

    if filtered_df.is_empty():
        return None

    if use_time_weighting:
        current_year = date.today().year

        df_with_years_ago = filtered_df.with_columns([
            (pl.lit(current_year) - pl.col(date_column).dt.year()).alias("years_ago")
        ])

        df_with_weight = df_with_years_ago.with_columns([
            (pl.lit(1.0) / (pl.col("years_ago") + 1)).alias("time_weight")
        ])
    else:
        df_with_weight = filtered_df.with_columns([
            pl.lit(1.0).alias("time_weight")
        ])

    grouped = df_with_weight.group_by(["route_key", "carrier_price"]).agg([
        pl.count().alias("volume"),
        pl.mean("time_weight").alias("avg_time_weight")
    ])

    grouped = grouped.with_columns([
        (pl.col("volume") * pl.col("avg_time_weight")).alias("adjusted_volume"),
        (pl.col("carrier_price") * pl.col("volume") * pl.col("avg_time_weight")).alias("adjusted_cost")
    ])

    stats = grouped.select([
        (pl.col("adjusted_cost").sum() / pl.col("adjusted_volume").sum()).alias("volume_time_weighted_avg_cost"),
        pl.col("carrier_price").mean().alias("avg_cost"),
        pl.col("carrier_price").median().alias("median_cost"),
        pl.col("carrier_price").min().alias("min_cost"),
        pl.col("carrier_price").max().alias("max_cost"),
        pl.col("carrier_price").std(ddof=1).alias("stddev_cost"),
        pl.col("volume").sum().alias("valid_count")
    ])

    return dict(zip(stats.columns, stats.row(0)))





def join_dfs(*dfs: pl.DataFrame) -> pl.DataFrame:
    """
    Combine multiple DataFrames by vertical stacking (not deduplicating), which increases volume count for repeated routes.
    """
    return pl.concat(dfs, how="vertical")


def estimate_price(
    matches_3: pl.DataFrame,
    within_10: pl.DataFrame,
    within_20: pl.DataFrame,
    matches_2: pl.DataFrame,
    min_count: int = 5,
    use_time_weighting: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Estimate pricing using a tiered fallback strategy. Joins data instead of concatenating unique rows to retain volume.
    """

    def try_tier(df: pl.DataFrame, label: str, use_time_weighting) -> Optional[Dict[str, Any]]:
        stats = estimate_price_from_df(df, use_time_weighting=use_time_weighting)
        if stats and stats["valid_count"] >= min_count:
            stats["explanation"] = f"Estimated from {stats['valid_count']} loads; ({label})."
            return stats
        return None

    # Tier 1
    result = try_tier(matches_3, "3-letter prefix", use_time_weighting)
    if result:
        return result

    # Tier 2
    result = try_tier(join_dfs(matches_3, within_10), "3-letter prefix + within 10km", use_time_weighting)
    if result:
        return result

    # Tier 3
    result = try_tier(join_dfs(matches_3, within_10, within_20), "3-letter prefix + within 10km + within 20km", use_time_weighting)
    if result:
        return result

    # Tier 4
    result = try_tier(join_dfs(matches_3, within_10, within_20, matches_2), "3-letter prefix + within 10km + within 20km + 2-letter prefix", use_time_weighting)
    if result:
        return result

    return None

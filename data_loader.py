import polars as pl
import pandas as pd
import pyodbc
import requests
from typing import Optional, Literal, Tuple


class DataLoader:
    def __init__(
        self,
        sql_server: Optional[str] = None,
        sql_database: Optional[str] = None,
        sql_username: Optional[str] = None,
        sql_password: Optional[str] = None,
        postcode_cache: Optional[dict[Tuple[float, float], str]] = None,
    ):
        self.connection_params = {
            "server": sql_server,
            "database": sql_database,
            "username": sql_username,
            "password": sql_password,
        }
        self.postcode_cache = postcode_cache or {}

    def load(self, source: Literal["parquet", "sql"], path: Optional[str] = None) -> pl.DataFrame:
        schema = {
            "origin_postcode": pl.Utf8,
            "origin_lat": pl.Float64,
            "origin_lon": pl.Float64,
            "destination_postcode": pl.Utf8,
            "dest_lat": pl.Float64,
            "dest_lon": pl.Float64,
            "vehicle_type": pl.Utf8,
            "pickup_date": pl.Datetime,
            "contract_type": pl.Utf8,
            "journey_distance": pl.Float64,
            "load_id": pl.Int64,
            "shipper_price": pl.Float64,
            "shipper_id": pl.Int64,
            "carrier_price": pl.Float64,
            "carrier_name": pl.Utf8,
        }

        if source == "parquet":
            if path is None:
                raise ValueError("You must provide a file path for parquet loading.")
            df = pl.read_parquet(path)
            df = df.with_columns(
                pl.col("origin_lat").cast(pl.Float64),
                pl.col("origin_lon").cast(pl.Float64),
                pl.col("dest_lat").cast(pl.Float64),
                pl.col("dest_lon").cast(pl.Float64),
            )

        elif source == "sql":
            df = self._load_from_sql(schema)
        else:
            raise ValueError("Source must be either 'parquet' or 'sql'.")

        return self._fill_missing_postcodes(df)

    def _load_from_sql(self, schema) -> pl.DataFrame:
        conn_str = (
            f"DRIVER={{ODBC Driver 18 for SQL Server}};"
            f"SERVER={self.connection_params['server']};"
            f"DATABASE={self.connection_params['database']};"
            f"UID={self.connection_params['username']};"
            f"PWD={self.connection_params['password']}"
        )
        query = """
        SELECT
          -- Mapped from loads.from
          lf.postcode           AS origin_postcode,
          lf.latitude           AS origin_lat,
          lf.longitude          AS origin_lon,
        
          -- Mapped from loads.to
          lt.postcode           AS destination_postcode,
          lt.latitude           AS dest_lat,
          lt.longitude          AS dest_lon,
        
          -- Mapped from loads.load
          ll.vehicleDisplayName AS vehicle_type,
          ll.collectBy          AS pickup_date,
          ll.jobDisplayDescription AS contract_type,
          ll.distance              AS journey_distance, 
          ll.loadId                AS load_id,
          ll.customerAgreedRate    as shipper_price,
          ll.customerContactId     AS shipper_id,
        
          -- Mapped from loads.order
          lo.agreedRate         AS carrier_price,
          lo.subcontractorName  AS carrier_name
        
        FROM loads.load ll
        JOIN loads."from" lf ON ll.loadId = lf.loadId
        JOIN loads."order" lo ON ll.loadId = lo.loadId
        JOIN loads."to" lt ON ll.loadId = lt.loadId
        WHERE lo.agreedRate IS NOT NULL
          AND lo.agreedRate > 0;
        """
        with pyodbc.connect(conn_str) as conn:
            # infer_schema_length, extends the read until it coerces the type
            df = pl.read_database(query, conn, schema_overrides=schema)
        return df

    def fill_postcodes_from_cache_polars(
        self,
        df: pl.DataFrame,
        lat_col: str,
        lon_col: str,
        postcode_col: str,
        rounding_dp: int = 2,
    ) -> pl.DataFrame:
        updated_postcodes = []
        for row in df.select([lat_col, lon_col, postcode_col]).iter_rows(named=True):
            current_postcode = row[postcode_col]

            if current_postcode is not None and len(current_postcode) > 3:
                updated_postcodes.append(current_postcode)
                continue

            lat, lon = row[lat_col], row[lon_col]

            if lat is None or lon is None:
                updated_postcodes.append(None)
                continue

            key = (round(lat, rounding_dp), round(lon, rounding_dp))
            updated_postcodes.append(self.postcode_cache.get(key, None))

        return df.with_columns(pl.Series(postcode_col, updated_postcodes))

    def _fill_missing_postcodes(self, df: pl.DataFrame) -> pl.DataFrame:
        LATITUDE_DECIMAL_PLACE_ROUNDING = 2

        df = self.fill_postcodes_from_cache_polars(
            df,
            lat_col="origin_lat",
            lon_col="origin_lon",
            postcode_col="origin_postcode",
            rounding_dp=LATITUDE_DECIMAL_PLACE_ROUNDING,
        )

        df = self.fill_postcodes_from_cache_polars(
            df,
            lat_col="dest_lat",
            lon_col="dest_lon",
            postcode_col="destination_postcode",
            rounding_dp=LATITUDE_DECIMAL_PLACE_ROUNDING,
        )

        return df

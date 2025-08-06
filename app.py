import streamlit as st
import polars as pl
from pathlib import Path
import pickle
from datetime import date, timedelta
from dotenv import load_dotenv
import os

from similar_routes import find_similar_routes_by_postcode
from price_estimator import estimate_price
from data_loader import DataLoader
from high_density_lane_price_estimator import identify_optimal_price
from plotting import display_results


@st.cache_data
def load_postcode_cache(postcode_cache_file_location):
    postcode_cache_file = Path(postcode_cache_file_location)
    if postcode_cache_file.exists():
        with open(postcode_cache_file_location, "rb") as handle:
            postcode_cache = pickle.load(handle)
    else:
        postcode_cache = {}
    return postcode_cache


postcode_cache = load_postcode_cache("postcode_cache.pickle")



@st.cache_data
def load_filtered_data(use_live_data=False):
    # Load environment variables from .env
    load_dotenv()
    
    # Read variables from environment
    sql_server = os.getenv("SQL_SERVER")
    sql_database = os.getenv("SQL_DATABASE")
    sql_username = os.getenv("SQL_USERNAME")
    sql_password = os.getenv("SQL_PASSWORD")
    
    # Optional: check if any are missing
    if not all([sql_server, sql_database, sql_username, sql_password]):
        raise ValueError("Missing one or more SQL environment variables.")
    
    # Instantiate the DataLoader with credentials
    loader = DataLoader(
        sql_server=sql_server,
        sql_database=sql_database,
        sql_username=sql_username,
        sql_password=sql_password,
        postcode_cache=postcode_cache
    )
    
    # Load data from SQL
    df = loader.load(source="sql")

    # data_loader = DataLoader(postcode_cache=postcode_cache)
    # df = data_loader.load(source="parquet", path="./analysis/fls_data.parquet")
    return df


# -- Load data --
df = load_filtered_data()

# -- Page UI --
st.title("Find Similar Routes")
st.markdown("Search for similar routes within a defined radius or postcode match.")

# -- Get unique vehicle types --
vehicle_types = ["All vehicle types"] + df.select("vehicle_type").unique().sort("vehicle_type").to_series().to_list()

# -- Input form --
with st.form("postcode_form"):
    origin_postcode = st.text_input("Origin Postcode", placeholder="e.g. SW1A 1AA")
    dest_postcode = st.text_input("Destination Postcode", placeholder="e.g. B1 1AA")
    selected_vehicle_type = st.selectbox("Vehicle Type", vehicle_types)
    submitted = st.form_submit_button("Find Similar Routes")



# -- On submit --
if submitted:
    if origin_postcode and dest_postcode:
        try:
            with st.spinner("Finding similar routes..."):
                try:
                    if selected_vehicle_type == "All vehicle types":
                        df_filtered = df
                    else:
                        df_filtered = df.filter(pl.col("vehicle_type") == selected_vehicle_type)

                    matches_3_letter, within_10km, within_20km, matches_2_letter = find_similar_routes_by_postcode(
                        df_filtered,
                        origin_postcode=origin_postcode,
                        dest_postcode=dest_postcode,
                    )

                except Exception as e:
                    st.exception(e)
                    st.stop()

            # Filter matches_3_letter for last 365 days
            one_year_ago = date.today() - timedelta(days=365)
            matches_3_letter_recent = matches_3_letter.filter(
                pl.col("pickup_date") >= one_year_ago
            )

            pricing_stats = None

            if matches_3_letter_recent.height >= 25:
                df_matches_pd = matches_3_letter_recent.to_pandas()
                optimal_price = identify_optimal_price(df_matches_pd)
                if optimal_price is not None:
                    pricing_stats = {
                        "volume_time_weighted_avg_cost": optimal_price,
                        "explanation": "Estimated using lowest fair price in last 365 days"
                    }

                display_results("3-letter postcode matches (last 365 days)", matches_3_letter_recent, price_estimate=optimal_price)

            elif matches_3_letter_recent.height > 0:
                st.error("Not enough recent 3-letter matches (minimum 25 in last 365 days) for reliable price estimate.")
                # display_results("3-letter postcode matches (last 365 days)", matches_3_letter)

                # display_results("Routes within 10 km", within_10km)
                # display_results("Routes within 20 km", within_20km)
                # display_results("2-letter postcode matches", matches_2_letter)

            else:
                st.warning("No data available")

            # Show price if available
            if pricing_stats is not None:
                st.success(f"Price Estimate: £{pricing_stats['volume_time_weighted_avg_cost']:.2f}")
                st.write(f"**Explanation:** {pricing_stats.get('explanation', '')}")

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter both origin and destination postcodes.")

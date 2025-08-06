import streamlit as st
import polars as pl
import plotly.express as px

def display_results(label: str, result_df: pl.DataFrame, price_estimate: float | None = None):
    st.subheader(f"{label} — {result_df.shape[0]} matches")

    if result_df.is_empty():
        st.info("No matches found.")
    else:
        #st.dataframe(result_df.to_pandas(), use_container_width=True)

        plot_df = result_df.select([
            "pickup_date", "carrier_price", "vehicle_type", "origin_postcode", "destination_postcode",
            "shipper_id", "carrier_name"
        ]).sort("pickup_date").to_pandas()

        plot_df["route"] = (
            plot_df["origin_postcode"].str.upper().str.strip() + " → " +
            plot_df["destination_postcode"].str.upper().str.strip()
        )

        # Convert for categorical treatment
        plot_df["shipper_id"] = plot_df["shipper_id"].astype(str)
        plot_df["carrier_name"] = plot_df["carrier_name"].astype(str)

        # Scatter plot by vehicle_type
        fig_vehicle = px.scatter(
            plot_df,
            x="pickup_date",
            y="carrier_price",
            color="vehicle_type",
            hover_name="route",
            title="Carrier Price over Time by Vehicle Type",
            labels={
                "pickup_date": "Pickup Date",
                "carrier_price": "Carrier Price (£)",
                "vehicle_type": "Vehicle Type"
            },
            opacity=0.8,
            template="plotly_white"
        )
        if price_estimate is not None:
            fig_vehicle.add_hline(
                y=price_estimate,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Estimated Price £{price_estimate:.2f}",
                annotation_position="top left"
            )
        st.plotly_chart(fig_vehicle, use_container_width=True, key=f"plot_vehicle_{label}")

        # Scatter plot by shipper_id
        shipper_ids_sorted = sorted(plot_df["shipper_id"].unique())
        fig_shipper = px.scatter(
            plot_df,
            x="pickup_date",
            y="carrier_price",
            color="shipper_id",
            category_orders={"shipper_id": shipper_ids_sorted},
            hover_name="route",
            title="Carrier Price over Time by Shipper ID",
            labels={
                "pickup_date": "Pickup Date",
                "carrier_price": "Carrier Price (£)",
                "shipper_id": "Shipper ID"
            },
            opacity=0.8,
            template="plotly_white"
        )
        fig_shipper.update_layout(
            legend_title_text='Shipper ID',
            legend=dict(
                itemsizing='constant',
                itemclick='toggleothers',
                itemdoubleclick='toggle'
            )
        )
        if price_estimate is not None:
            fig_shipper.add_hline(
                y=price_estimate,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Estimated Price £{price_estimate:.2f}",
                annotation_position="top left"
            )
        st.plotly_chart(fig_shipper, use_container_width=True, key=f"plot_shipper_{label}")

        # Scatter plot by carrier_name
        carrier_names_sorted = sorted(plot_df["carrier_name"].unique())
        fig_carrier = px.scatter(
            plot_df,
            x="pickup_date",
            y="carrier_price",
            color="carrier_name",
            category_orders={"carrier_name": carrier_names_sorted},
            hover_name="route",
            title="Carrier Price over Time by Carrier Name",
            labels={
                "pickup_date": "Pickup Date",
                "carrier_price": "Carrier Price (£)",
                "carrier_name": "Carrier Name"
            },
            opacity=0.8,
            template="plotly_white"
        )
        fig_carrier.update_layout(
            legend_title_text='Carrier Name',
            legend=dict(
                itemsizing='constant',
                itemclick='toggleothers',
                itemdoubleclick='toggle'
            )
        )
        if price_estimate is not None:
            fig_carrier.add_hline(
                y=price_estimate,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Estimated Price £{price_estimate:.2f}",
                annotation_position="top left"
            )
        st.plotly_chart(fig_carrier, use_container_width=True, key=f"plot_carrier_{label}")

        # --- Histogram with price overlay ---
        hist_fig = px.histogram(
            plot_df,
            x="carrier_price",
            nbins=max(1, int((plot_df["carrier_price"].max() - plot_df["carrier_price"].min()) / 10)),
            title="Carrier Price Distribution (£10 bins)",
            labels={"carrier_price": "Carrier Price (£)"},
            template="plotly_white"
        )
        if price_estimate is not None:
            hist_fig.add_vline(
                x=price_estimate,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Estimated Price £{price_estimate:.2f}",
                annotation_position="top left"
            )
        st.plotly_chart(hist_fig, use_container_width=True, key=f"hist_{label}")

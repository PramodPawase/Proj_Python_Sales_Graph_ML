"""Business supplier intelligence dashboard with profit/loss and demand forecasting."""

from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

from generate_dummy_data import build_dataset


st.set_page_config(page_title="SupplyLens | Business intelligence", page_icon=":material/query_stats:", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "business_orders.csv"
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@st.cache_data(ttl="1h")
def load_orders() -> pd.DataFrame:
    """Read and enrich the dummy order ledger."""
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(exist_ok=True)
        build_dataset().to_csv(DATA_PATH, index=False)
    data = pd.read_csv(DATA_PATH, parse_dates=["order_date"])
    data["gross_profit"] = data["sales_revenue"] - data["purchase_cost"] - data["shipping_cost"] - data["operating_cost"]
    data["profit_margin"] = np.where(data["sales_revenue"] > 0, data["gross_profit"] / data["sales_revenue"], 0)
    data["month"] = data["order_date"].dt.to_period("M").dt.to_timestamp()
    data["month_name"] = data["order_date"].dt.month.map(dict(enumerate(MONTH_NAMES, 1)))
    data["year"] = data["order_date"].dt.year
    return data


@st.cache_data(ttl="1h")
def forecast_monthly_demand(data: pd.DataFrame) -> pd.DataFrame:
    """Train a small regression model to forecast the next 12 monthly quantities."""
    monthly = data.groupby("month", as_index=False).agg(quantity=("quantity", "sum"), orders=("order_count", "sum"))
    monthly["month_number"] = monthly["month"].dt.month
    monthly["time_index"] = np.arange(len(monthly))
    X = monthly[["time_index", "month_number"]]
    model = RandomForestRegressor(n_estimators=250, max_depth=5, random_state=42)
    model.fit(X, monthly["quantity"])
    future_months = pd.date_range(monthly["month"].max() + pd.offsets.MonthBegin(1), periods=12, freq="MS")
    future = pd.DataFrame({"month": future_months})
    future["month_number"] = future["month"].dt.month
    future["time_index"] = np.arange(len(monthly), len(monthly) + 12)
    future["forecast_quantity"] = model.predict(future[["time_index", "month_number"]]).round().astype(int)
    baseline = monthly["quantity"].mean()
    future["recommendation"] = np.select(
        [future["forecast_quantity"] < baseline * 0.85, future["forecast_quantity"] > baseline * 1.15],
        ["Reduce supplier allocation", "Increase supplier allocation"],
        default="Maintain supplier allocation",
    )
    return future


def money(value: float) -> str:
    return f"₹{value:,.0f}"


orders = load_orders()

st.title("SupplyLens business intelligence")
st.caption("Profit and loss analysis, supplier performance, demand seasonality, and data-driven supply actions.")

with st.sidebar:
    st.header("Analysis filters")
    min_date, max_date = orders["order_date"].min().date(), orders["order_date"].max().date()
    selected_dates = st.date_input("Order date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    all_suppliers = sorted(orders["supplier_name"].unique())
    selected_suppliers = st.multiselect("Suppliers", all_suppliers, default=all_suppliers)
    all_categories = sorted(orders["category"].unique())
    selected_categories = st.multiselect("Product categories", all_categories, default=all_categories)
    st.download_button("Download dataset", DATA_PATH.read_bytes(), file_name="business_orders.csv", mime="text/csv", icon=":material/download:")

if len(selected_dates) != 2:
    st.info("Choose both a start and end date to show analysis.")
    st.stop()

start_date, end_date = pd.Timestamp(selected_dates[0]), pd.Timestamp(selected_dates[1])
filtered = orders[
    orders["order_date"].between(start_date, end_date)
    & orders["supplier_name"].isin(selected_suppliers)
    & orders["category"].isin(selected_categories)
].copy()

if filtered.empty:
    st.warning("No matching orders. Broaden the filters to continue.")
    st.stop()

monthly_kpis = filtered.groupby("month", as_index=False).agg(revenue=("sales_revenue", "sum"), profit=("gross_profit", "sum"))
profit = filtered["gross_profit"].sum()
revenue = filtered["sales_revenue"].sum()
total_cost = filtered[["purchase_cost", "shipping_cost", "operating_cost"]].sum().sum()
margin = profit / revenue if revenue else 0

with st.container(horizontal=True):
    st.metric("Revenue", money(revenue), border=True, chart_data=monthly_kpis["revenue"].tolist(), chart_type="line")
    st.metric("Profit / loss", money(profit), border=True, chart_data=monthly_kpis["profit"].tolist(), chart_type="bar")
    st.metric("Profit margin", f"{margin:.1%}", border=True)
    st.metric("Outstanding payments", money(filtered["pending_amount"].sum()), border=True)
    st.metric("Purchase orders", f"{int(filtered['order_count'].sum()):,}", border=True)

overview_tab, suppliers_tab, seasonal_tab, data_tab = st.tabs(["Overview", "Supplier intelligence", "Seasonal action plan", "Order ledger"])

with overview_tab:
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.subheader("Monthly profit and loss")
            pl = monthly_kpis.melt("month", var_name="Measure", value_name="Amount")
            chart = alt.Chart(pl).mark_line(point=True).encode(
                x=alt.X("month:T", title="Month"), y=alt.Y("Amount:Q", title="Amount (₹)"),
                color=alt.Color("Measure:N", scale=alt.Scale(domain=["revenue", "profit"], range=["#3274A1", "#1B9E77"])),
                tooltip=[alt.Tooltip("month:T", title="Month"), alt.Tooltip("Measure:N"), alt.Tooltip("Amount:Q", format=",.0f")],
            ).properties(height=320)
            st.altair_chart(chart)
    with right:
        with st.container(border=True):
            st.subheader("Cost composition")
            costs = pd.DataFrame({"Cost type": ["Purchase", "Shipping", "Operating"], "Amount": [filtered["purchase_cost"].sum(), filtered["shipping_cost"].sum(), filtered["operating_cost"].sum()]})
            st.altair_chart(alt.Chart(costs).mark_arc(innerRadius=55).encode(theta=alt.Theta("Amount:Q"), color=alt.Color("Cost type:N"), tooltip=["Cost type:N", alt.Tooltip("Amount:Q", format=",.0f")]).properties(height=320))
    with st.container(border=True):
        st.subheader("Most profitable products")
        product_profit = filtered.groupby(["product_name", "category"], as_index=False).agg(revenue=("sales_revenue", "sum"), profit=("gross_profit", "sum"), quantity=("quantity", "sum")).sort_values("profit", ascending=False).head(8)
        st.altair_chart(alt.Chart(product_profit).mark_bar().encode(x=alt.X("profit:Q", title="Profit (₹)"), y=alt.Y("product_name:N", sort="-x", title=None), color=alt.Color("category:N"), tooltip=["product_name:N", "category:N", alt.Tooltip("profit:Q", format=",.0f"), "quantity:Q"]).properties(height=280))

with suppliers_tab:
    supplier_summary = filtered.groupby("supplier_name", as_index=False).agg(
        orders=("order_count", "sum"), quantity=("quantity", "sum"), spend=("purchase_cost", "sum"), revenue=("sales_revenue", "sum"), profit=("gross_profit", "sum"), pending=("pending_amount", "sum"), avg_delivery_days=("delivery_days", "mean"), reliability=("supplier_reliability", "mean"),
    )
    supplier_summary["margin"] = supplier_summary["profit"] / supplier_summary["revenue"]
    supplier_summary["supplier_status"] = np.select(
        [supplier_summary["orders"] >= supplier_summary["orders"].quantile(0.75), supplier_summary["orders"] <= supplier_summary["orders"].quantile(0.25)],
        ["High order volume", "Low order volume"], default="Medium order volume",
    )
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.subheader("Who receives the most orders?")
            st.altair_chart(alt.Chart(supplier_summary).mark_bar().encode(x=alt.X("orders:Q", title="Total orders"), y=alt.Y("supplier_name:N", sort="-x", title=None), color=alt.Color("supplier_status:N", scale=alt.Scale(domain=["High order volume", "Medium order volume", "Low order volume"], range=["#1B9E77", "#3274A1", "#E76F51"])), tooltip=["supplier_name:N", "orders:Q", "quantity:Q"]).properties(height=330))
    with right:
        with st.container(border=True):
            st.subheader("Supplier profitability and delivery")
            st.altair_chart(alt.Chart(supplier_summary).mark_circle(opacity=0.85).encode(x=alt.X("avg_delivery_days:Q", title="Average delivery days"), y=alt.Y("margin:Q", title="Profit margin", axis=alt.Axis(format="%")), size=alt.Size("spend:Q", title="Purchase spend"), color=alt.Color("supplier_name:N", legend=None), tooltip=["supplier_name:N", "orders:Q", alt.Tooltip("margin:Q", format=".1%"), alt.Tooltip("spend:Q", format=",.0f"), alt.Tooltip("pending:Q", format=",.0f")]).properties(height=330))
    st.subheader("Supplier scorecard")
    scorecard = supplier_summary.sort_values("orders", ascending=False)[["supplier_name", "supplier_status", "orders", "quantity", "spend", "profit", "margin", "pending", "avg_delivery_days", "reliability"]]
    st.dataframe(scorecard, hide_index=True, column_config={"spend": st.column_config.NumberColumn("Purchase spend", format="₹%d"), "profit": st.column_config.NumberColumn("Profit", format="₹%d"), "pending": st.column_config.NumberColumn("Pending", format="₹%d"), "margin": st.column_config.NumberColumn("Margin", format="percent"), "reliability": st.column_config.ProgressColumn("Reliability", min_value=0, max_value=1, format="%.0f%%")})

with seasonal_tab:
    forecast = forecast_monthly_demand(orders)
    st.subheader("ML demand forecast and supplier allocation guidance")
    st.caption("A Random Forest model learns the monthly demand pattern from order history. Recommendations compare forecast demand with the historic monthly average.")
    fc_chart = alt.Chart(forecast).mark_bar().encode(x=alt.X("month:T", title="Forecast month"), y=alt.Y("forecast_quantity:Q", title="Expected quantity"), color=alt.Color("recommendation:N", scale=alt.Scale(domain=["Increase supplier allocation", "Maintain supplier allocation", "Reduce supplier allocation"], range=["#1B9E77", "#3274A1", "#E76F51"])), tooltip=[alt.Tooltip("month:T", title="Month"), "forecast_quantity:Q", "recommendation:N"]).properties(height=350)
    st.altair_chart(fc_chart)
    st.subheader("When should supply be reduced?")
    reduce_months = forecast[forecast["recommendation"] == "Reduce supplier allocation"]
    if reduce_months.empty:
        st.success("The forecast has no low-demand months below the reduction threshold.")
    else:
        st.warning("Reduce or renegotiate supplier allocation for: " + ", ".join(reduce_months["month"].dt.strftime("%B %Y")) + ".")
    st.dataframe(forecast.assign(month=forecast["month"].dt.strftime("%b %Y")), hide_index=True, column_config={"forecast_quantity": st.column_config.NumberColumn("Forecast quantity", format="%d")})

with data_tab:
    st.subheader("Filtered order ledger")
    display = filtered.sort_values("order_date", ascending=False)[["order_id", "order_date", "supplier_name", "product_name", "category", "quantity", "order_count", "sales_revenue", "purchase_cost", "gross_profit", "payment_mode", "pending_amount"]]
    st.dataframe(display, hide_index=True, column_config={"order_date": st.column_config.DateColumn("Order date", format="DD MMM YYYY"), "sales_revenue": st.column_config.NumberColumn("Revenue", format="₹%d"), "purchase_cost": st.column_config.NumberColumn("Purchase cost", format="₹%d"), "gross_profit": st.column_config.NumberColumn("Profit", format="₹%d"), "pending_amount": st.column_config.NumberColumn("Pending", format="₹%d")})

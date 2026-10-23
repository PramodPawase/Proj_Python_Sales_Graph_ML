"""Forecasting and purchase-recommendation functions for SupplyLens."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


TARGET_COLUMNS = ["quantity", "revenue", "profit"]
FEATURE_COLUMNS = [
    "time_index", "month_number", "month_sin", "month_cos",
    "quantity_lag_1", "quantity_average_3", "revenue_lag_1",
    "revenue_average_3", "profit_lag_1", "profit_average_3",
]


def monthly_series(data: pd.DataFrame, supplier: str | None = None, product: str | None = None) -> pd.DataFrame:
    """Return a continuous monthly history for one supplier-product combination."""
    frame = data.copy()
    if supplier is not None:
        frame = frame[frame["supplier_name"] == supplier]
    if product is not None:
        frame = frame[frame["product_name"] == product]
    monthly = frame.groupby("month", as_index=False).agg(
        quantity=("quantity", "sum"), revenue=("sales_revenue", "sum"), profit=("gross_profit", "sum"),
    )
    if monthly.empty:
        return monthly
    all_months = pd.date_range(monthly["month"].min(), monthly["month"].max(), freq="MS")
    return monthly.set_index("month").reindex(all_months, fill_value=0).rename_axis("month").reset_index()


def add_features(monthly: pd.DataFrame) -> pd.DataFrame:
    """Build lagged and seasonal features without using future target values."""
    features = monthly.copy().sort_values("month").reset_index(drop=True)
    features["time_index"] = np.arange(len(features))
    features["month_number"] = features["month"].dt.month
    features["month_sin"] = np.sin(2 * np.pi * features["month_number"] / 12)
    features["month_cos"] = np.cos(2 * np.pi * features["month_number"] / 12)
    for target in TARGET_COLUMNS:
        features[f"{target}_lag_1"] = features[target].shift(1)
        features[f"{target}_average_3"] = features[target].shift(1).rolling(3).mean()
    return features.dropna().reset_index(drop=True)


def _future_feature(history: pd.DataFrame, month: pd.Timestamp, time_index: int) -> dict[str, float]:
    result: dict[str, float] = {
        "time_index": time_index, "month_number": month.month,
        "month_sin": float(np.sin(2 * np.pi * month.month / 12)),
        "month_cos": float(np.cos(2 * np.pi * month.month / 12)),
    }
    for target in TARGET_COLUMNS:
        result[f"{target}_lag_1"] = float(history[target].iloc[-1])
        result[f"{target}_average_3"] = float(history[target].tail(3).mean())
    return result


def forecast_one_series(monthly: pd.DataFrame, periods: int = 12) -> tuple[pd.DataFrame, dict[str, float]]:
    """Fit a multi-output Random Forest and forecast quantity, revenue, and profit."""
    train = add_features(monthly)
    if len(train) < 8:
        raise ValueError("At least eight complete months are required for forecasting.")
    validation_months = min(3, max(1, len(train) // 4))
    train_set, validation_set = train.iloc[:-validation_months], train.iloc[-validation_months:]
    validation_model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    validation_model.fit(train_set[FEATURE_COLUMNS], train_set[TARGET_COLUMNS])
    validation_prediction = validation_model.predict(validation_set[FEATURE_COLUMNS])
    validation_mae = {target: float(mean_absolute_error(validation_set[target], validation_prediction[:, index])) for index, target in enumerate(TARGET_COLUMNS)}
    model = RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42)
    model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMNS])
    history = monthly[TARGET_COLUMNS].copy().reset_index(drop=True)
    future_months = pd.date_range(monthly["month"].max() + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
    forecasts: list[dict[str, object]] = []
    for offset, month in enumerate(future_months, start=1):
        feature = _future_feature(history, month, len(monthly) + offset - 1)
        quantity, revenue, profit = np.maximum(model.predict(pd.DataFrame([feature], columns=FEATURE_COLUMNS))[0], 0)
        forecasts.append({"month": month, "forecast_quantity": round(quantity), "forecast_revenue": round(revenue, 2), "forecast_profit": round(profit, 2)})
        history.loc[len(history)] = [quantity, revenue, profit]
    return pd.DataFrame(forecasts), validation_mae


def forecast_all_products(data: pd.DataFrame, periods: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forecast every supplier/product series and return validation accuracy."""
    forecasts: list[pd.DataFrame] = []
    accuracy_rows: list[dict[str, object]] = []
    for row in data[["supplier_name", "product_name", "category"]].drop_duplicates().itertuples(index=False):
        series = monthly_series(data, row.supplier_name, row.product_name)
        forecast, validation_mae = forecast_one_series(series, periods)
        forecast[["supplier_name", "product_name", "category"]] = [row.supplier_name, row.product_name, row.category]
        forecasts.append(forecast)
        accuracy_rows.append({"supplier_name": row.supplier_name, "product_name": row.product_name, "quantity_mae": validation_mae["quantity"], "revenue_mae": validation_mae["revenue"], "profit_mae": validation_mae["profit"]})
    return pd.concat(forecasts, ignore_index=True), pd.DataFrame(accuracy_rows)


def create_purchase_plan(data: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    """Create a next-month plan from demand forecast, safety stock, and stock on hand."""
    plan = forecast[forecast["month"] == forecast["month"].min()].copy()
    latest_stock = data.sort_values("order_date").groupby(["supplier_name", "product_name"], as_index=False).tail(1)
    plan = plan.merge(latest_stock[["supplier_name", "product_name", "current_stock", "reorder_point", "unit_cost", "supplier_reliability", "delivery_days"]], on=["supplier_name", "product_name"], how="left")
    plan["safety_stock"] = np.ceil(plan["forecast_quantity"] * 0.20).astype(int)
    plan["recommended_order_quantity"] = np.maximum(0, plan["forecast_quantity"] + plan["safety_stock"] - plan["current_stock"]).astype(int)
    plan["estimated_purchase_cost"] = plan["recommended_order_quantity"] * plan["unit_cost"]
    plan["allocation_action"] = np.select([(plan["recommended_order_quantity"] == 0) | (plan["forecast_quantity"] < plan["reorder_point"] * 0.85), plan["recommended_order_quantity"] > plan["reorder_point"]], ["Reduce / hold order", "Increase allocation"], default="Maintain allocation")
    return plan.sort_values(["allocation_action", "estimated_purchase_cost"], ascending=[True, False])

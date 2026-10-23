"""Create a reproducible dummy business-orders dataset for the dashboard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_PATH = Path(__file__).parent / "data" / "business_orders.csv"


def build_dataset() -> pd.DataFrame:
    """Generate two years of supplier purchase and customer sales records."""
    rng = np.random.default_rng(20261023)
    suppliers = [
        ("Apex Electronics", "Electronics", 0.88, 1.00),
        ("BrightHome Supply", "Home & Kitchen", 0.78, 0.82),
        ("Cedar Office Works", "Office", 0.72, 0.68),
        ("Delta Retail Partners", "Personal Care", 0.83, 0.58),
        ("Evergreen Wholesale", "Food & Beverage", 0.68, 0.94),
        ("Fusion Tech Traders", "Electronics", 0.91, 0.49),
    ]
    products = {
        "Electronics": [("Wireless Earbuds", 850, 1399), ("Smart Watch", 1650, 2599), ("USB-C Charger", 290, 549)],
        "Home & Kitchen": [("Electric Kettle", 640, 1099), ("Storage Container Set", 370, 699), ("LED Desk Lamp", 520, 949)],
        "Office": [("Notebook Pack", 125, 249), ("Ergonomic Mouse", 480, 899), ("Printer Paper Box", 310, 559)],
        "Personal Care": [("Skin Care Kit", 620, 1099), ("Hair Dryer", 790, 1399), ("Electric Trimmer", 690, 1199)],
        "Food & Beverage": [("Premium Coffee Pack", 310, 569), ("Healthy Snack Box", 250, 449), ("Tea Collection", 280, 519)],
    }
    monthly_seasonality = {
        1: 0.70, 2: 0.74, 3: 0.88, 4: 0.92, 5: 0.82, 6: 0.76,
        7: 0.79, 8: 0.86, 9: 0.96, 10: 1.15, 11: 1.38, 12: 1.48,
    }
    payment_modes = ["UPI", "Bank Transfer", "Credit", "Cash"]
    rows: list[dict[str, object]] = []
    order_number = 1001
    for month_start in pd.date_range("2025-01-01", "2026-10-01", freq="MS"):
        for supplier, category, reliability, supplier_share in suppliers:
            product_name, cost_price, sales_price = products[category][rng.integers(0, 3)]
            seasonality = monthly_seasonality[month_start.month]
            category_factor = 1.20 if category == "Food & Beverage" and month_start.month in (10, 11, 12) else 1.0
            order_count = max(1, int(round((7 + 18 * supplier_share) * seasonality * category_factor + rng.normal(0, 1.3))))
            quantity = max(8, int(round(order_count * rng.uniform(3.0, 6.2))))
            date = month_start + pd.Timedelta(days=int(rng.integers(0, 27)))
            discount = rng.uniform(0.00, 0.12)
            unit_cost = round(cost_price * rng.uniform(0.94, 1.08), 2)
            unit_price = round(sales_price * (1 - discount), 2)
            revenue = round(quantity * unit_price, 2)
            purchase_cost = round(quantity * unit_cost, 2)
            shipping_cost = round(purchase_cost * rng.uniform(0.025, 0.07), 2)
            operating_cost = round(revenue * rng.uniform(0.035, 0.09), 2)
            payment_mode = payment_modes[rng.choice(len(payment_modes), p=[0.34, 0.34, 0.25, 0.07])]
            credit_ratio = rng.uniform(0.10, 0.45) if payment_mode == "Credit" else rng.uniform(0, 0.06)
            pending_amount = round(revenue * credit_ratio, 2)
            delivery_days = max(1, int(round(10 - reliability * 5 + rng.normal(0, 1.5))))
            current_stock = int(round(quantity * rng.uniform(0.55, 1.35)))
            reorder_point = int(round(quantity * 0.75))
            rows.append({
                "order_id": f"ORD-{order_number}",
                "order_date": date.date().isoformat(),
                "supplier_name": supplier,
                "product_name": product_name,
                "category": category,
                "unit_cost": unit_cost,
                "unit_selling_price": unit_price,
                "quantity": quantity,
                "order_count": order_count,
                "purchase_cost": purchase_cost,
                "sales_revenue": revenue,
                "shipping_cost": shipping_cost,
                "operating_cost": operating_cost,
                "payment_mode": payment_mode,
                "pending_amount": pending_amount,
                "delivery_days": delivery_days,
                "supplier_reliability": reliability,
                "current_stock": current_stock,
                "reorder_point": reorder_point,
            })
            order_number += 1
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    build_dataset().to_csv(OUTPUT_PATH, index=False)
    print(f"Created {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

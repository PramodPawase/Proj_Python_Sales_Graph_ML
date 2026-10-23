# SupplyLens business intelligence dashboard

A Streamlit dashboard for a business owner to monitor profit/loss, compare supplier order volume, predict product-level sales, and plan supplier purchasing.

## Included data

`data/business_orders.csv` is a reproducible dummy dataset covering January 2025 through October 2026. It includes order dates, supplier and product details, selling and purchase prices, quantities, payment modes, pending amounts, delivery days, reliability, stock on hand, and reorder points.

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

To recreate the CSV with the same seeded dummy data:

```powershell
python generate_dummy_data.py
```

## Analysis included

- Revenue, total costs, profit/loss, margin, and outstanding payments
- Top profitable products and monthly revenue vs. profit trend
- Supplier order-volume ranking, profitability, delivery time, and pending payments
- High/medium/low supplier order classification
- Product-and-supplier Random Forest forecasts for quantity, revenue, and profit
- Lag, three-month average, and seasonal ML features
- Validation MAE for quantity, revenue, and profit predictions
- Next-month purchase plan using forecast demand, safety stock, and current stock

# SupplyLens business intelligence dashboard

A Streamlit dashboard for a business owner to monitor profit/loss, compare supplier order volume, identify low-volume suppliers, and use a small ML demand forecast to plan supplier allocation.

## Included data

`data/business_orders.csv` is a reproducible dummy dataset covering January 2025 through October 2026. It includes order dates, supplier and product details, selling and purchase prices, quantities, order counts, payment modes, pending amounts, delivery days, and reliability scores.

## Run locally

```powershell
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
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
- Random Forest forecast for the next 12 months and supplier allocation recommendations

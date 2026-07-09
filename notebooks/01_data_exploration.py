"""
01 - Exploratory Data Analysis
Dunnhumby: The Complete Journey

Run after: python src/data_pipeline.py
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Load Data ──────────────────────────────────────────────
print("Loading processed data...")
master = pd.read_parquet(PROCESSED_DIR / "master_transactions.parquet")
rfm = pd.read_parquet(PROCESSED_DIR / "rfm_table.parquet")

print(f"Master transactions: {master.shape}")
print(f"Households (RFM):    {rfm.shape}")
print(f"Date range: Week {master['WEEK_NO'].min()} to {master['WEEK_NO'].max()}")
print(f"Unique households:   {master['household_key'].nunique()}")
print(f"Unique products:     {master['PRODUCT_ID'].nunique()}")
print(f"Unique departments:  {master['DEPARTMENT'].nunique()}")


# ── 1. Revenue Over Time ──────────────────────────────────
weekly_rev = master.groupby("WEEK_NO").agg(
    revenue=("SALES_VALUE", "sum"),
    transactions=("BASKET_ID", "nunique"),
    avg_basket=("SALES_VALUE", "mean"),
).reset_index()

fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=("Weekly Revenue", "Weekly Transaction Count"),
    shared_xaxes=True,
)
fig.add_trace(
    go.Scatter(x=weekly_rev["WEEK_NO"], y=weekly_rev["revenue"],
               mode="lines", name="Revenue", line=dict(color="#2196F3")),
    row=1, col=1,
)
fig.add_trace(
    go.Scatter(x=weekly_rev["WEEK_NO"], y=weekly_rev["transactions"],
               mode="lines", name="Transactions", line=dict(color="#4CAF50")),
    row=2, col=1,
)
fig.update_layout(title="Revenue & Transaction Trends (102 Weeks)", height=600, width=1000)
fig.write_html(str(OUTPUT_DIR / "01_revenue_trends.html"))
print("\nSaved: revenue trends")


# ── 2. Top Departments ────────────────────────────────────
dept_rev = master.groupby("DEPARTMENT")["SALES_VALUE"].sum().nlargest(10).reset_index()
fig = px.bar(
    dept_rev, x="SALES_VALUE", y="DEPARTMENT", orientation="h",
    title="Top 10 Departments by Revenue",
    labels={"SALES_VALUE": "Total Revenue ($)", "DEPARTMENT": ""},
    color_discrete_sequence=["#2196F3"],
)
fig.update_layout(yaxis=dict(autorange="reversed"), height=400, width=800)
fig.write_html(str(OUTPUT_DIR / "01_top_departments.html"))
print("Saved: top departments")


# ── 3. Top Product Categories ─────────────────────────────
cat_rev = master.groupby("COMMODITY_DESC")["SALES_VALUE"].sum().nlargest(15).reset_index()
fig = px.bar(
    cat_rev, x="SALES_VALUE", y="COMMODITY_DESC", orientation="h",
    title="Top 15 Product Categories by Revenue",
    labels={"SALES_VALUE": "Total Revenue ($)", "COMMODITY_DESC": ""},
    color_discrete_sequence=["#FF9800"],
)
fig.update_layout(yaxis=dict(autorange="reversed"), height=500, width=800)
fig.write_html(str(OUTPUT_DIR / "01_top_categories.html"))
print("Saved: top categories")


# ── 4. Discount Distribution ──────────────────────────────
promo_stats = master.groupby("ON_PROMO")["SALES_VALUE"].agg(["sum", "count"]).reset_index()
promo_stats.columns = ["On Promo", "Revenue", "Transactions"]
promo_stats["On Promo"] = promo_stats["On Promo"].map({0: "Full Price", 1: "Discounted"})

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Revenue Split", "Transaction Split"),
    specs=[[{"type": "pie"}, {"type": "pie"}]],
)
fig.add_trace(
    go.Pie(labels=promo_stats["On Promo"], values=promo_stats["Revenue"],
           marker=dict(colors=["#4CAF50", "#FF5722"])),
    row=1, col=1,
)
fig.add_trace(
    go.Pie(labels=promo_stats["On Promo"], values=promo_stats["Transactions"],
           marker=dict(colors=["#4CAF50", "#FF5722"])),
    row=1, col=2,
)
fig.update_layout(title="Full Price vs Discounted Purchases", height=400, width=800)
fig.write_html(str(OUTPUT_DIR / "01_promo_split.html"))
print("Saved: promo split")


# ── 5. Discount Depth Distribution ────────────────────────
promo_only = master[master["ON_PROMO"] == 1]
fig = px.histogram(
    promo_only, x="DISCOUNT_DEPTH", nbins=50,
    title="Distribution of Discount Depth (when on promotion)",
    labels={"DISCOUNT_DEPTH": "Discount as % of Shelf Price"},
    color_discrete_sequence=["#9C27B0"],
)
fig.update_layout(height=400, width=800)
fig.write_html(str(OUTPUT_DIR / "01_discount_depth_dist.html"))
print("Saved: discount depth distribution")


# ── 6. RFM Distributions ──────────────────────────────────
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("Recency (weeks since last purchase)", "Frequency (unique visits)", "Monetary (total spend)"),
)
fig.add_trace(go.Histogram(x=rfm["recency"], nbinsx=30, marker_color="#2196F3"), row=1, col=1)
fig.add_trace(go.Histogram(x=rfm["frequency"], nbinsx=30, marker_color="#4CAF50"), row=1, col=2)
fig.add_trace(go.Histogram(x=rfm["monetary"], nbinsx=30, marker_color="#FF9800"), row=1, col=3)
fig.update_layout(title="RFM Distributions", height=350, width=1100, showlegend=False)
fig.write_html(str(OUTPUT_DIR / "01_rfm_distributions.html"))
print("Saved: RFM distributions")


# ── 7. Promo Channel Usage ────────────────────────────────
channel_data = pd.DataFrame({
    "Channel": ["Retail Discount", "Coupon", "In-Store Display", "Mailer"],
    "Pct Transactions": [
        master["HAS_RETAIL_DISC"].mean() * 100,
        master["HAS_COUPON_DISC"].mean() * 100,
        master["HAD_DISPLAY"].mean() * 100,
        master["HAD_MAILER"].mean() * 100,
    ],
})
fig = px.bar(
    channel_data, x="Channel", y="Pct Transactions",
    title="% of Transactions Exposed to Each Promo Channel",
    color_discrete_sequence=["#00BCD4"],
)
fig.update_layout(height=400, width=600)
fig.write_html(str(OUTPUT_DIR / "01_promo_channels.html"))
print("Saved: promo channels")


# ── 8. Price Variation Across Categories ──────────────────
top_cats = master.groupby("COMMODITY_DESC")["SALES_VALUE"].sum().nlargest(10).index
price_data = master[master["COMMODITY_DESC"].isin(top_cats)]

fig = px.box(
    price_data, x="COMMODITY_DESC", y="UNIT_PRICE",
    title="Price Distribution: Top 10 Categories",
    labels={"COMMODITY_DESC": "Category", "UNIT_PRICE": "Unit Price ($)"},
)
fig.update_layout(height=500, width=1000, xaxis_tickangle=-45)
fig.write_html(str(OUTPUT_DIR / "01_price_variation.html"))
print("Saved: price variation")


# ── Summary Stats ─────────────────────────────────────────
print("\n" + "=" * 60)
print("EDA SUMMARY")
print("=" * 60)
print(f"Total revenue:              ${master['SALES_VALUE'].sum():,.0f}")
print(f"Avg basket value:           ${master.groupby('BASKET_ID')['SALES_VALUE'].sum().mean():,.2f}")
print(f"% transactions on promo:    {master['ON_PROMO'].mean() * 100:.1f}%")
print(f"Avg discount depth (promo): {promo_only['DISCOUNT_DEPTH'].mean() * 100:.1f}%")
print(f"Median household spend:     ${rfm['monetary'].median():,.0f}")
print(f"Median visit frequency:     {rfm['frequency'].median():.0f} visits")
print(f"Top department:             {dept_rev.iloc[0]['DEPARTMENT']}")
print(f"Top category:               {cat_rev.iloc[0]['COMMODITY_DESC']}")
print("=" * 60)
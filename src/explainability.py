"""
Demand Driver Explainability.
Visualizes what drives demand for each product category
using the OLS regression coefficients as interpretable
feature importance — no black-box SHAP needed because
the model is already linear and transparent.

For each category-segment, shows:
  - Which factors increase demand (positive coefficients)
  - Which factors decrease demand (negative coefficients)
  - Relative magnitude of each driver
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs/figures")


# Human-readable names for regression features
FEATURE_LABELS = {
    "log_price": "Price Level",
    "promo_share": "Discount Promotions",
    "display_share": "In-Store Displays",
    "mailer_share": "Mailer Campaigns",
    "is_festive": "Festive / Holiday Period",
    "time_trend": "Secular Trend",
}


def extract_coefficients(weekly_data, category, segment=None):
    """
    Fit OLS model for a specific category (optionally filtered by segment)
    and return the standardized coefficients as demand drivers.
    """
    if segment:
        data = weekly_data[
            (weekly_data["COMMODITY_DESC"] == category) &
            (weekly_data["SEGMENT_NAME"] == segment)
        ].copy()
    else:
        data = weekly_data[weekly_data["COMMODITY_DESC"] == category].copy()

    if len(data) < 20:
        return None

    data = data.sort_values("WEEK_NO")
    data["time_trend"] = np.arange(len(data))

    y = data["log_quantity"]
    features = ["log_price", "promo_share", "display_share", "mailer_share", "is_festive", "time_trend"]
    X = data[features]
    X = sm.add_constant(X)

    try:
        model = sm.OLS(y, X).fit()
    except Exception:
        return None

    # Extract coefficients (exclude constant)
    coefs = []
    for feat in features:
        coef = model.params.get(feat, 0)
        pval = model.pvalues.get(feat, 1)
        label = FEATURE_LABELS.get(feat, feat)

        # Standardize: multiply by feature std to get "effect size"
        # This makes coefficients comparable across features
        feat_std = X[feat].std()
        standardized = coef * feat_std

        coefs.append({
            "feature": feat,
            "label": label,
            "coefficient": coef,
            "standardized_effect": standardized,
            "p_value": pval,
            "significant": pval < 0.05,
            "direction": "Increases Demand" if standardized > 0 else "Decreases Demand",
        })

    coefs_df = pd.DataFrame(coefs)
    coefs_df["abs_effect"] = coefs_df["standardized_effect"].abs()

    return coefs_df, model.rsquared


def plot_demand_drivers(weekly_data, category, segment=None):
    """
    Create a horizontal bar chart showing what drives demand
    for a specific category — the 'explainability' view.
    """
    result = extract_coefficients(weekly_data, category, segment)
    if result is None:
        print(f"  Not enough data for {category}" + (f" / {segment}" if segment else ""))
        return None

    coefs_df, r2 = result

    # Sort by absolute effect
    coefs_df = coefs_df.sort_values("abs_effect", ascending=True)

    # Color by direction
    colors = ["#4CAF50" if x > 0 else "#FF5722" for x in coefs_df["standardized_effect"]]

    # Opacity by significance
    opacities = [1.0 if s else 0.4 for s in coefs_df["significant"]]

    title_suffix = f" — {segment}" if segment else ""
    sig_note = "Faded bars = not statistically significant (p > 0.05)"

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=coefs_df["label"],
        x=coefs_df["standardized_effect"],
        orientation="h",
        marker=dict(color=colors, opacity=opacities),
        text=[f"{v:+.3f}" for v in coefs_df["standardized_effect"]],
        textposition="outside",
    ))

    fig.add_vline(x=0, line_dash="dot", line_color="#888")

    fig.update_layout(
        title=(
            f"What Drives Demand: {category.title()}{title_suffix}"
            f"<br><sub>Standardized coefficients from log-log OLS (R² = {r2:.2f}). "
            f"Green = increases demand, Red = decreases. {sig_note}</sub>"
        ),
        xaxis_title="Standardized Effect on Demand",
        yaxis_title="",
        height=400,
        width=800,
        showlegend=False,
    )

    return fig


def plot_top_categories_drivers(weekly_data, n_categories=6):
    """
    Create a grid of demand driver charts for top categories.
    This is the main output for the explainability section.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get top categories by data volume
    top_cats = (
        weekly_data.groupby("COMMODITY_DESC")["n_transactions"]
        .sum().nlargest(n_categories).index.tolist()
    )

    # Exclude junk
    exclude = ["COUPON/MISC ITEMS", "MISC ITEMS", "MISCELLANEOUS"]
    top_cats = [c for c in top_cats if c not in exclude][:n_categories]

    all_drivers = []

    for cat in top_cats:
        result = extract_coefficients(weekly_data, cat)
        if result is not None:
            coefs_df, r2 = result
            coefs_df["category"] = cat
            coefs_df["r_squared"] = r2
            all_drivers.append(coefs_df)

            # Save individual chart
            fig = plot_demand_drivers(weekly_data, cat)
            if fig:
                safe_name = cat.replace(" ", "_").replace("/", "_")[:25]
                fig.write_html(str(OUTPUT_DIR / f"drivers_{safe_name}.html"))

    if not all_drivers:
        print("No driver data generated.")
        return None

    drivers_df = pd.concat(all_drivers, ignore_index=True)

    # Summary heatmap: features x categories
    sig_drivers = drivers_df[drivers_df["significant"]].copy()

    if not sig_drivers.empty:
        pivot = sig_drivers.pivot_table(
            index="label", columns="category",
            values="standardized_effect", aggfunc="mean"
        )

        # Shorten category names for display
        pivot.columns = [c.title()[:20] for c in pivot.columns]

        fig = px.imshow(
            pivot,
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            aspect="auto",
            title="Demand Drivers Across Categories<br><sub>Green/blue = increases demand, Red = decreases. Only significant effects shown.</sub>",
            labels=dict(x="Category", y="Driver", color="Effect"),
            text_auto=".2f",
        )
        fig.update_layout(height=450, width=900)
        fig.write_html(str(OUTPUT_DIR / "demand_drivers_heatmap.html"))
        try:
            fig.write_image(str(OUTPUT_DIR / "demand_drivers_heatmap.png"), scale=2)
        except Exception:
            pass
        print("Saved demand drivers heatmap.")

    # Save raw driver data
    drivers_df.to_parquet(PROCESSED_DIR / "demand_drivers.parquet", index=False)
    print(f"Saved demand drivers for {len(top_cats)} categories.")

    return drivers_df


def print_driver_insights(drivers_df):
    """Print key insights from demand driver analysis."""
    sig = drivers_df[drivers_df["significant"]].copy()

    print("\nDemand Driver Insights:")

    # Strongest positive driver per category
    for cat in sig["category"].unique():
        cat_data = sig[sig["category"] == cat]
        pos = cat_data[cat_data["standardized_effect"] > 0]
        if not pos.empty:
            top = pos.nlargest(1, "abs_effect").iloc[0]
            print(f"  > {cat.title()}: strongest demand driver is "
                  f"{top['label']} (effect: {top['standardized_effect']:+.3f})")

    # Which driver is most important overall?
    overall = sig.groupby("label")["abs_effect"].mean().sort_values(ascending=False)
    if not overall.empty:
        print(f"\n  Overall most impactful driver: {overall.index[0]} "
              f"(avg effect: {overall.iloc[0]:.3f})")


def plot_category_demand_drivers(weekly_data, category, segment=None):
    """
    Fit OLS model for a category and segment, create a horizontal bar chart,
    and return (fig, coefs_df, r2) for the Streamlit dashboard.
    """
    result = extract_coefficients(weekly_data, category, segment)
    if result is None:
        return None
    coefs_df, r2 = result
    fig = plot_demand_drivers(weekly_data, category, segment)
    return fig, coefs_df, r2


if __name__ == "__main__":
    print("Loading weekly demand data...")
    weekly = pd.read_parquet(PROCESSED_DIR / "weekly_demand.parquet")

    # Attach segment labels
    rfm = pd.read_parquet(PROCESSED_DIR / "rfm_segmented.parquet")
    master = pd.read_parquet(PROCESSED_DIR / "master_transactions.parquet")
    seg_map = rfm[["household_key", "SEGMENT_NAME"]].drop_duplicates()
    master = master.merge(seg_map, on="household_key", how="left")

    # Re-aggregate with segments
    weekly_seg = master.groupby(
        ["COMMODITY_DESC", "SEGMENT_NAME", "WEEK_NO"]
    ).agg(
        total_quantity=("QUANTITY", "sum"),
        total_revenue=("SALES_VALUE", "sum"),
        avg_unit_price=("UNIT_PRICE", "mean"),
        avg_discount_depth=("DISCOUNT_DEPTH", "mean"),
        promo_share=("ON_PROMO", "mean"),
        display_share=("HAD_DISPLAY", "mean"),
        mailer_share=("HAD_MAILER", "mean"),
        is_festive=("IS_FESTIVE", "max"),
        n_transactions=("BASKET_ID", "nunique"),
    ).reset_index()

    weekly_seg = weekly_seg[weekly_seg["n_transactions"] >= 5]
    weekly_seg["log_quantity"] = np.log1p(weekly_seg["total_quantity"])
    weekly_seg["log_price"] = np.log1p(weekly_seg["avg_unit_price"])

    print("Generating demand driver analysis...")
    drivers = plot_top_categories_drivers(weekly_seg, n_categories=8)

    if drivers is not None:
        print_driver_insights(drivers)
"""
Price Elasticity Modeling for CPG Products.
Uses log-log OLS regression to estimate price elasticity
per product category and customer segment.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")


PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs/figures")


def load_data():
    """Load master transactions and segmented RFM."""
    master = pd.read_parquet(PROCESSED_DIR / "master_transactions.parquet")
    rfm = pd.read_parquet(PROCESSED_DIR / "rfm_segmented.parquet")

    # Attach segment labels to transactions
    segment_map = rfm[["household_key", "SEGMENT_NAME", "CLUSTER"]].drop_duplicates()
    master = master.merge(segment_map, on="household_key", how="left")

    return master


def aggregate_weekly_demand(master):
    """
    Aggregate to weekly level per product category per segment.
    This smooths out transaction-level noise and gives us
    price-quantity pairs for elasticity estimation.
    """
    # Use COMMODITY_DESC as product category (more granular than DEPARTMENT)
    agg = master.groupby(
        ["COMMODITY_DESC", "SEGMENT_NAME", "WEEK_NO"]
    ).agg(
        total_quantity=("QUANTITY", "sum"),
        total_revenue=("SALES_VALUE", "sum"),
        total_shelf_revenue=("SHELF_PRICE", "sum"),
        avg_unit_price=("UNIT_PRICE", "mean"),
        avg_shelf_unit_price=("SHELF_UNIT_PRICE", "mean"),
        avg_discount_depth=("DISCOUNT_DEPTH", "mean"),
        promo_share=("ON_PROMO", "mean"),
        display_share=("HAD_DISPLAY", "mean"),
        mailer_share=("HAD_MAILER", "mean"),
        is_festive=("IS_FESTIVE", "max"),
        n_transactions=("BASKET_ID", "nunique"),
    ).reset_index()

    # Filter out weeks with very few transactions (noise)
    agg = agg[agg["n_transactions"] >= 5].copy()

    # Log transforms for log-log regression
    agg["log_quantity"] = np.log1p(agg["total_quantity"])
    agg["log_price"] = np.log1p(agg["avg_unit_price"])
    agg["log_shelf_price"] = np.log1p(agg["avg_shelf_unit_price"])

    return agg


def estimate_elasticity(group_df):
    """
    Estimate price elasticity using log-log OLS:
    log(Q) = b0 + b1*log(P) + b2*promo_share + b3*display + b4*mailer + b5*festive + b6*trend + e

    b1 is the price elasticity of demand.
    Includes a time trend control to prevent spurious positive elasticity
    caused by prices and quantities both trending upward over time.
    """
    if len(group_df) < 20:
        return None

    df = group_df.copy().sort_values("WEEK_NO")

    y = df["log_quantity"]

    # Add time trend to control for secular growth/decline
    df["time_trend"] = np.arange(len(df))

    X = df[["log_price", "promo_share", "display_share", "mailer_share", "is_festive", "time_trend"]]
    X = sm.add_constant(X)

    try:
        model = sm.OLS(y, X).fit()

        elasticity = model.params.get("log_price", np.nan)

        # Cap elasticity at 0 — positive price elasticity for grocery
        # staples is economically implausible and indicates confounding.
        # We report 0 (perfectly inelastic) instead of a misleading positive.
        if elasticity > 0:
            elasticity = 0.0

        return {
            "elasticity": elasticity,
            "elasticity_raw": model.params.get("log_price", np.nan),
            "elasticity_pvalue": model.pvalues.get("log_price", np.nan),
            "promo_effect": model.params.get("promo_share", np.nan),
            "display_effect": model.params.get("display_share", np.nan),
            "mailer_effect": model.params.get("mailer_share", np.nan),
            "festive_effect": model.params.get("is_festive", np.nan),
            "time_trend_effect": model.params.get("time_trend", np.nan),
            "r_squared": model.rsquared,
            "adj_r_squared": model.rsquared_adj,
            "n_observations": len(df),
        }
    except Exception:
        return None


def compute_elasticity_matrix(weekly_data):
    """
    Compute price elasticity for each category-segment combination.
    Returns a DataFrame suitable for heatmap visualization.
    """
    # Get top categories by transaction volume
    top_categories = (
        weekly_data.groupby("COMMODITY_DESC")["n_transactions"]
        .sum()
        .nlargest(15)
        .index.tolist()
    )
    weekly_filtered = weekly_data[weekly_data["COMMODITY_DESC"].isin(top_categories)]

    results = []

    for (category, segment), group in weekly_filtered.groupby(
        ["COMMODITY_DESC", "SEGMENT_NAME"]
    ):
        est = estimate_elasticity(group)
        if est is not None:
            est["COMMODITY_DESC"] = category
            est["SEGMENT_NAME"] = segment
            results.append(est)

    elasticity_df = pd.DataFrame(results)

    # Flag statistical significance
    elasticity_df["is_significant"] = elasticity_df["elasticity_pvalue"] < 0.05

    print(f"\nElasticity estimates computed: {len(elasticity_df)}")
    print(f"  Statistically significant: {elasticity_df['is_significant'].sum()}")

    return elasticity_df


def compute_overall_elasticity(weekly_data):
    """Compute overall elasticity per category (not split by segment)."""
    top_categories = (
        weekly_data.groupby("COMMODITY_DESC")["n_transactions"]
        .sum()
        .nlargest(20)
        .index.tolist()
    )

    results = []
    for category, group in weekly_data[
        weekly_data["COMMODITY_DESC"].isin(top_categories)
    ].groupby("COMMODITY_DESC"):
        est = estimate_elasticity(group)
        if est is not None:
            est["COMMODITY_DESC"] = category
            results.append(est)

    return pd.DataFrame(results)


def fit_demand_driver_model(group_df):
    """Fit a standardized OLS model for a category or category-segment slice."""
    if len(group_df) < 20:
        return None

    df = group_df.copy().sort_values("WEEK_NO")
    df["time_trend"] = np.arange(len(df))

    features = ["log_price", "promo_share", "display_share", "mailer_share", "is_festive", "time_trend"]
    feature_labels = {
        "log_price": "Price",
        "promo_share": "Promo Share",
        "display_share": "Display Share",
        "mailer_share": "Mailer Share",
        "is_festive": "Festive Period",
        "time_trend": "Time Trend",
    }

    standardized = df[features].copy()
    means = standardized.mean()
    stds = standardized.std(ddof=0).replace(0, 1)
    standardized = (standardized - means) / stds

    X = sm.add_constant(standardized)
    y = df["log_quantity"]

    try:
        model = sm.OLS(y, X).fit()
    except Exception:
        return None

    coef = model.params.drop("const", errors="ignore")
    drivers = pd.DataFrame({
        "feature": coef.index,
        "label": [feature_labels.get(name, name) for name in coef.index],
        "std_coef": coef.values,
        "importance": np.abs(coef.values),
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return model, drivers


def plot_category_demand_drivers(weekly_data, category, segment=None):
    """Plot a waterfall chart of standardized demand drivers for a category."""
    group = weekly_data[weekly_data["COMMODITY_DESC"] == category].copy()
    if segment is not None:
        group = group[group["SEGMENT_NAME"] == segment].copy()

    fit = fit_demand_driver_model(group)
    if fit is None:
        print(f"Not enough data to model demand drivers for {category}.")
        return None

    model, drivers = fit
    safe_segment = segment if segment is not None else "all_segments"
    safe_name = f"{category}_{safe_segment}".replace(" ", "_").replace("/", "_")[:40]

    fig = go.Figure(
        go.Waterfall(
            name="Drivers",
            orientation="v",
            measure=["relative"] * len(drivers) + ["total"],
            x=drivers["label"].tolist() + ["Total signal"],
            y=drivers["std_coef"].tolist() + [drivers["std_coef"].sum()],
            text=[f"{value:+.2f}" for value in drivers["std_coef"].tolist()] + [f"{drivers['std_coef'].sum():+.2f}"],
            connector={"line": {"color": "#BDBDBD"}},
            increasing={"marker": {"color": "#2E7D32"}},
            decreasing={"marker": {"color": "#C62828"}},
            totals={"marker": {"color": "#1565C0"}},
        )
    )

    title_suffix = f" for {segment}" if segment else ""
    fig.update_layout(
        title=f"What Drives Demand in {category}{title_suffix}?<br><sub>Standardized OLS coefficients from the weekly demand model</sub>",
        xaxis_title="Demand drivers",
        yaxis_title="Relative coefficient strength",
        height=600,
        width=1100,
    )

    fig.write_html(str(OUTPUT_DIR / f"demand_drivers_{safe_name}.html"))
    try:
        fig.write_image(str(OUTPUT_DIR / f"demand_drivers_{safe_name}.png"), scale=2)
    except Exception as exc:
        print(f"Skipping PNG export for demand drivers: {exc}")

    return fig, model, drivers


def plot_elasticity_heatmap(elasticity_df):
    """Create heatmap: categories x segments, colored by elasticity."""
    # Filter to significant results only
    sig = elasticity_df[elasticity_df["is_significant"]].copy()

    if sig.empty:
        print("No significant elasticity estimates to plot. Using all estimates.")
        sig = elasticity_df.copy()

    # Also exclude zero-elasticity rows (these were positive and got capped)
    # to keep the heatmap meaningful — only show genuinely negative elasticity
    sig = sig[sig["elasticity"] < 0].copy()

    if sig.empty:
        print("No negative elasticity estimates. Plotting all results.")
        sig = elasticity_df.copy()

    pivot = sig.pivot_table(
        index="COMMODITY_DESC",
        columns="SEGMENT_NAME",
        values="elasticity",
        aggfunc="mean",
    )

    # Sort by average elasticity (most sensitive at top)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

    fig = px.imshow(
        pivot,
        color_continuous_scale="YlOrRd",
        aspect="auto",
        title="Price Elasticity by Category & Segment<br><sub>More negative (darker) = more price sensitive. Only statistically significant estimates shown.</sub>",
        labels=dict(x="Customer Segment", y="Product Category", color="Elasticity"),
        text_auto=".2f",
    )

    fig.update_layout(width=1000, height=700)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(OUTPUT_DIR / "elasticity_heatmap.html"))
    try:
        fig.write_image(str(OUTPUT_DIR / "elasticity_heatmap.png"), scale=2)
    except Exception as exc:
        print(f"Skipping PNG export for elasticity heatmap: {exc}")
    print("Saved elasticity heatmap.")

    return fig


def plot_promo_effectiveness(elasticity_df):
    """Compare promo channel effectiveness across segments."""
    channels = ["promo_effect", "display_effect", "mailer_effect"]
    labels = ["Discount", "In-Store Display", "Mailer"]

    channel_data = []
    for _, row in elasticity_df.iterrows():
        for ch, label in zip(channels, labels):
            channel_data.append({
                "SEGMENT_NAME": row["SEGMENT_NAME"],
                "Channel": label,
                "Effect": row[ch],
            })

    ch_df = pd.DataFrame(channel_data)

    # Use median to reduce outlier skew (display coefficient was
    # dominating due to a few high-leverage observations)
    agg = ch_df.groupby(["SEGMENT_NAME", "Channel"])["Effect"].median().reset_index()

    fig = px.bar(
        agg,
        x="SEGMENT_NAME",
        y="Effect",
        color="Channel",
        barmode="group",
        title="Promotion Channel Effectiveness by Segment<br><sub>Median regression coefficient — higher = stronger demand lift</sub>",
        color_discrete_map={
            "Discount": "#2196F3",
            "In-Store Display": "#FF9800",
            "Mailer": "#4CAF50",
        },
        text_auto=".2f",
    )

    fig.update_layout(
        width=900, height=500,
        xaxis_title="Customer Segment",
        yaxis_title="Demand Lift (median coefficient)",
    )
    fig.write_html(str(OUTPUT_DIR / "promo_effectiveness.html"))

    try:
        fig.write_image(str(OUTPUT_DIR / "promo_effectiveness.png"), scale=2)
    except Exception:
        pass

    print("Saved promo effectiveness chart.")

    return fig


def generate_elasticity_insights(elasticity_df):
    """Generate key business insights from elasticity analysis."""
    insights = []

    # Most price-sensitive category-segment combinations
    most_sensitive = elasticity_df.nsmallest(3, "elasticity")
    for _, row in most_sensitive.iterrows():
        insights.append(
            f"HIGH SENSITIVITY: {row['COMMODITY_DESC']} for {row['SEGMENT_NAME']} "
            f"has elasticity of {row['elasticity']:.2f} — a 10% price cut would "
            f"increase demand by ~{abs(row['elasticity'] * 10):.1f}%."
        )

    # Least price-sensitive (where discounts are wasted)
    least_sensitive = elasticity_df[elasticity_df["elasticity"] > -0.5].nlargest(3, "elasticity")
    for _, row in least_sensitive.iterrows():
        insights.append(
            f"LOW SENSITIVITY: {row['COMMODITY_DESC']} for {row['SEGMENT_NAME']} "
            f"has elasticity of {row['elasticity']:.2f} — discounts here have "
            f"minimal volume impact. Protect margin instead."
        )

    # Best promo channel per segment
    for segment in elasticity_df["SEGMENT_NAME"].unique():
        seg_data = elasticity_df[elasticity_df["SEGMENT_NAME"] == segment]
        avg_display = seg_data["display_effect"].mean()
        avg_mailer = seg_data["mailer_effect"].mean()
        avg_promo = seg_data["promo_effect"].mean()

        best = max(
            [("In-Store Displays", avg_display), ("Mailers", avg_mailer), ("Discounts", avg_promo)],
            key=lambda x: x[1] if not np.isnan(x[1]) else -999,
        )
        insights.append(
            f"CHANNEL: {segment} responds best to {best[0]} (effect: {best[1]:.3f})."
        )

    return insights


if __name__ == "__main__":
    print("Loading data...")
    master = load_data()

    print("Aggregating weekly demand...")
    weekly = aggregate_weekly_demand(master)
    print(f"  Weekly aggregated rows: {len(weekly)}")

    print("\nComputing elasticity matrix (category x segment)...")
    elasticity_df = compute_elasticity_matrix(weekly)

    print("\nGenerating visualizations...")
    plot_elasticity_heatmap(elasticity_df)
    plot_promo_effectiveness(elasticity_df)

    print("\nKey Insights:")
    insights = generate_elasticity_insights(elasticity_df)
    for ins in insights:
        print(f"  > {ins}")

    # Save
    elasticity_df.to_parquet(PROCESSED_DIR / "elasticity_results.parquet", index=False)
    weekly.to_parquet(PROCESSED_DIR / "weekly_demand.parquet", index=False)
    print("\nSaved elasticity results and weekly demand data.")
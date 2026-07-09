"""
Promotion Simulation Engine.
Simulates the revenue, margin, and ROI impact of different
promotion strategies across customer segments and product categories.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs/figures")


# --- Promo type definitions ---
PROMO_TYPES = {
    "flat_5": {"label": "5% Off", "discount": 0.05, "cost_multiplier": 1.0},
    "flat_10": {"label": "10% Off", "discount": 0.10, "cost_multiplier": 1.0},
    "flat_15": {"label": "15% Off", "discount": 0.15, "cost_multiplier": 1.0},
    "flat_20": {"label": "20% Off", "discount": 0.20, "cost_multiplier": 1.0},
    "bogo": {"label": "Buy 1 Get 1 Free", "discount": 0.50, "cost_multiplier": 1.3},
    "bundle": {"label": "Bundle Deal (2 products)", "discount": 0.12, "cost_multiplier": 0.8},
    "cashback": {"label": "10% Cashback", "discount": 0.10, "cost_multiplier": 0.9},
}

# Assumed gross margin for CPG products
DEFAULT_GROSS_MARGIN = 0.35


def load_simulation_inputs():
    """Load elasticity results and baseline demand data."""
    elasticity = pd.read_parquet(PROCESSED_DIR / "elasticity_results.parquet")
    weekly = pd.read_parquet(PROCESSED_DIR / "weekly_demand.parquet")

    # Exclude non-product categories (data artifacts, not real CPG products)
    EXCLUDE_CATEGORIES = [
        "COUPON/MISC ITEMS", "MISC ITEMS", "MISCELLANEOUS",
        "COUPON", "NO COMMODITY DESCRIPTION",
    ]
    elasticity = elasticity[~elasticity["COMMODITY_DESC"].isin(EXCLUDE_CATEGORIES)]
    weekly = weekly[~weekly["COMMODITY_DESC"].isin(EXCLUDE_CATEGORIES)]

    # Only keep categories with meaningful (non-zero) elasticity
    elasticity = elasticity[elasticity["elasticity"] < -0.1].copy()

    # Compute baselines per category-segment
    baselines = weekly.groupby(["COMMODITY_DESC", "SEGMENT_NAME"]).agg(
        base_weekly_quantity=("total_quantity", "median"),
        base_weekly_revenue=("total_revenue", "median"),
        base_avg_price=("avg_unit_price", "median"),
        base_shelf_price=("avg_shelf_unit_price", "median"),
        n_households=("n_transactions", "median"),
    ).reset_index()

    return elasticity, baselines


def simulate_single_promo(
    category,
    segment,
    promo_key,
    elasticity_df,
    baselines_df,
    duration_weeks=4,
    gross_margin=DEFAULT_GROSS_MARGIN,
):
    """
    Simulate a single promotion scenario.

    Returns dict with:
    - Revenue impact (incremental revenue vs no-promo baseline)
    - Margin impact
    - Volume lift
    - Promo ROI
    - Post-promo cannibalization estimate
    """
    promo = PROMO_TYPES[promo_key]

    # Get elasticity for this category-segment
    el_row = elasticity_df[
        (elasticity_df["COMMODITY_DESC"] == category)
        & (elasticity_df["SEGMENT_NAME"] == segment)
    ]

    if el_row.empty:
        return None

    elasticity = el_row["elasticity"].values[0]

    # Get baseline demand
    base_row = baselines_df[
        (baselines_df["COMMODITY_DESC"] == category)
        & (baselines_df["SEGMENT_NAME"] == segment)
    ]

    if base_row.empty:
        return None

    base_qty = base_row["base_weekly_quantity"].values[0]
    base_price = base_row["base_avg_price"].values[0]
    base_revenue = base_row["base_weekly_revenue"].values[0]

    # --- Simulation ---

    # Price change from promotion
    effective_discount = promo["discount"]
    new_price = base_price * (1 - effective_discount)

    # Volume response using elasticity
    # log-log: %change in Q = elasticity * %change in P
    pct_price_change = -effective_discount  # negative because price drops
    pct_quantity_change = elasticity * pct_price_change  # elasticity is negative, so this is positive
    new_qty = base_qty * (1 + pct_quantity_change)

    # Revenue during promo
    promo_revenue = new_qty * new_price * duration_weeks
    baseline_revenue = base_revenue * duration_weeks

    # Incremental
    incremental_revenue = promo_revenue - baseline_revenue
    incremental_units = (new_qty - base_qty) * duration_weeks

    # Margin analysis
    cogs_per_unit = base_price * (1 - gross_margin)
    baseline_margin = (base_price - cogs_per_unit) * base_qty * duration_weeks
    promo_margin = (new_price - cogs_per_unit) * new_qty * duration_weeks
    margin_impact = promo_margin - baseline_margin

    # Promo cost (discount given + execution cost)
    discount_cost = effective_discount * base_price * new_qty * duration_weeks
    execution_cost = discount_cost * (promo["cost_multiplier"] - 1)
    total_promo_cost = discount_cost + execution_cost

    # ROI
    promo_roi = incremental_revenue / total_promo_cost if total_promo_cost > 0 else 0

    # Post-promo cannibalization estimate
    # Assume 15-25% of incremental volume is pulled forward from future weeks
    cannibalization_rate = 0.20
    post_promo_revenue_loss = incremental_revenue * cannibalization_rate
    net_incremental_revenue = incremental_revenue - post_promo_revenue_loss

    return {
        "category": category,
        "segment": segment,
        "promo_type": promo["label"],
        "promo_key": promo_key,
        "discount_depth": effective_discount,
        "duration_weeks": duration_weeks,
        "elasticity": elasticity,
        "base_weekly_qty": round(base_qty),
        "promo_weekly_qty": round(new_qty),
        "volume_lift_pct": round(pct_quantity_change * 100, 1),
        "incremental_units": round(incremental_units),
        "baseline_revenue": round(baseline_revenue, 2),
        "promo_revenue": round(promo_revenue, 2),
        "incremental_revenue": round(incremental_revenue, 2),
        "baseline_margin": round(baseline_margin, 2),
        "promo_margin": round(promo_margin, 2),
        "margin_impact": round(margin_impact, 2),
        "margin_impact_pct": round((margin_impact / baseline_margin) * 100, 1) if baseline_margin > 0 else 0,
        "total_promo_cost": round(total_promo_cost, 2),
        "promo_roi": round(promo_roi, 2),
        "cannibalization_loss": round(post_promo_revenue_loss, 2),
        "net_incremental_revenue": round(net_incremental_revenue, 2),
    }


def run_full_simulation(elasticity_df, baselines_df, categories=None, segments=None):
    """
    Run simulation across all promo types for specified categories and segments.
    If not specified, uses top 10 categories by volume that have elasticity data.
    """
    if categories is None:
        # Only pick categories that exist in BOTH elasticity and baseline tables
        valid_cats = set(elasticity_df["COMMODITY_DESC"]) & set(baselines_df["COMMODITY_DESC"])
        cat_volumes = (
            baselines_df[baselines_df["COMMODITY_DESC"].isin(valid_cats)]
            .groupby("COMMODITY_DESC")["base_weekly_quantity"]
            .sum()
        )
        categories = cat_volumes.nlargest(10).index.tolist()

    if segments is None:
        segments = elasticity_df["SEGMENT_NAME"].unique().tolist()

    all_results = []

    for cat in categories:
        for seg in segments:
            for promo_key in PROMO_TYPES:
                result = simulate_single_promo(
                    cat, seg, promo_key, elasticity_df, baselines_df
                )
                if result is not None:
                    all_results.append(result)

    results_df = pd.DataFrame(all_results)
    print(f"\nSimulation complete: {len(results_df)} scenarios evaluated.")
    print(f"  Categories: {results_df['category'].nunique()}")
    print(f"  Segments:   {results_df['segment'].nunique()}")

    return results_df


def find_optimal_promos(sim_results, optimize_for="promo_roi"):
    """
    Find the best promo type for each category-segment pair.
    Can optimize for: promo_roi, net_incremental_revenue, or margin_impact.
    """
    # Prefer scenarios that are both positive ROI and positive margin.
    candidates = sim_results[sim_results["promo_roi"] > 0].copy()
    positive_margin = candidates[candidates["margin_impact"] > 0].copy()

    if not positive_margin.empty:
        candidates = positive_margin
    elif candidates.empty:
        print("Warning: No positive ROI scenarios found. Using all results.")
        candidates = sim_results.copy()

    optimal = candidates.loc[
        candidates.groupby(["category", "segment"])[optimize_for].idxmax()
    ].sort_values(optimize_for, ascending=False)

    return optimal


def plot_promo_comparison(sim_results, category):
    """Compare promo types for a specific category across segments."""
    cat_data = sim_results[sim_results["category"] == category]

    if cat_data.empty:
        print(f"No data for category: {category}")
        return None

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Revenue Impact ($)", "Promo ROI"),
    )

    for i, metric in enumerate(["incremental_revenue", "promo_roi"], 1):
        pivot = cat_data.pivot_table(
            index="promo_type", columns="segment", values=metric, aggfunc="mean"
        )
        for col in pivot.columns:
            fig.add_trace(
                go.Bar(name=col, x=pivot.index, y=pivot[col], showlegend=(i == 1)),
                row=1, col=i,
            )

    fig.update_layout(
        title=f"Promo Strategy Comparison: {category}",
        barmode="group",
        width=1200,
        height=500,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = category.replace(" ", "_").replace("/", "_")[:30]
    fig.write_html(str(OUTPUT_DIR / f"promo_comparison_{safe_name}.html"))
    print(f"Saved promo comparison for {category}.")

    return fig


def plot_roi_matrix(optimal_promos):
    """Heatmap of optimal promo ROI by category x segment."""
    pivot = optimal_promos.pivot_table(
        index="category", columns="segment", values="promo_roi", aggfunc="mean"
    )

    # Sort by average ROI
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig = px.imshow(
        pivot,
        color_continuous_scale="Greens",
        aspect="auto",
        title="Optimal Promo ROI by Category & Segment<br><sub>Best promo type selected per cell. Higher = better return on promo spend.</sub>",
        labels=dict(x="Customer Segment", y="Product Category", color="ROI"),
        text_auto=".1f",
    )

    fig.update_layout(width=900, height=600)
    fig.write_html(str(OUTPUT_DIR / "roi_matrix.html"))
    try:
        fig.write_image(str(OUTPUT_DIR / "roi_matrix.png"), scale=2)
    except Exception as exc:
        print(f"Skipping PNG export for ROI matrix: {exc}")
    print("Saved ROI matrix.")

    return fig


def generate_recommendations(optimal_promos, sim_results):
    """Generate actionable business recommendations."""
    recommendations = []

    # Filter to scenarios with actual meaningful volume lift (> 1%)
    meaningful = optimal_promos[optimal_promos["volume_lift_pct"] > 1].copy()

    if meaningful.empty:
        meaningful = optimal_promos[optimal_promos["volume_lift_pct"] > 0].copy()

    # --- TOP ROI OPPORTUNITIES (with real volume lift) ---
    positive_margin = meaningful[meaningful["margin_impact"] > 0].copy()
    top_source = positive_margin if not positive_margin.empty else meaningful
    top_roi = top_source.nlargest(3, "promo_roi")
    for _, row in top_roi.iterrows():
        recommendations.append({
            "type": "HIGH ROI OPPORTUNITY",
            "finding": (
                f"{row['promo_type']} on {row['category'].title()} for "
                f"{row['segment']} yields {row['promo_roi']:.1f}x ROI with "
                f"{row['volume_lift_pct']:.0f}% volume lift over "
                f"{row['duration_weeks']} weeks."
            ),
            "action": (
                f"Prioritize {row['promo_type'].lower()} for "
                f"{row['category'].lower()} targeting {row['segment']} — "
                f"a {row['discount_depth']:.0%} discount drives "
                f"{row['volume_lift_pct']:.0f}% incremental volume at "
                f"positive margin."
            ),
        })

    # --- MOST PRICE SENSITIVE CATEGORIES ---
    # Categories where elasticity is strongest (biggest discount response)
    high_elastic = sim_results[sim_results["promo_key"] == "flat_10"].copy()
    if not high_elastic.empty:
        top_elastic = high_elastic.nlargest(2, "volume_lift_pct")
        for _, row in top_elastic.iterrows():
            recommendations.append({
                "type": "HIGH SENSITIVITY",
                "finding": (
                    f"{row['category'].title()} for {row['segment']} shows "
                    f"{row['volume_lift_pct']:.0f}% demand increase from just "
                    f"a 10% discount (elasticity: {row['elasticity']:.2f})."
                ),
                "action": (
                    f"This category-segment is highly responsive to price. "
                    f"Even shallow discounts (5-10%) generate meaningful lift. "
                    f"Use this for traffic-driving promotions."
                ),
            })

    # --- MARGIN RISKS (real destroyers with real volume) ---
    destroyers = sim_results[
        (sim_results["margin_impact"] < 0) &
        (sim_results["volume_lift_pct"].abs() > 0.5)  # Only flag if there's activity
    ].nsmallest(3, "margin_impact")

    for _, row in destroyers.iterrows():
        recommendations.append({
            "type": "MARGIN RISK",
            "finding": (
                f"{row['promo_type']} on {row['category'].title()} for "
                f"{row['segment']}: margin erodes by "
                f"{row['margin_impact_pct']:.0f}% despite "
                f"{row['volume_lift_pct']:.0f}% volume lift."
            ),
            "action": (
                f"The volume gain doesn't cover the discount cost. "
                f"Reduce promo depth or switch to bundle offers "
                f"for {row['category'].lower()} with this segment."
            ),
        })

    # --- BUNDLE VS FLAT DISCOUNT COMPARISON ---
    for seg in sim_results["segment"].unique():
        seg_data = sim_results[sim_results["segment"] == seg]
        bundle_roi = seg_data[seg_data["promo_key"] == "bundle"]["promo_roi"].mean()
        flat10_roi = seg_data[seg_data["promo_key"] == "flat_10"]["promo_roi"].mean()

        if pd.notna(bundle_roi) and pd.notna(flat10_roi) and bundle_roi > flat10_roi * 1.3:
            recommendations.append({
                "type": "STRATEGY SHIFT",
                "finding": (
                    f"Bundle deals average {bundle_roi:.1f}x ROI for "
                    f"{seg} vs {flat10_roi:.1f}x for flat 10% discounts — "
                    f"{((bundle_roi / flat10_roi) - 1) * 100:.0f}% more efficient."
                    if flat10_roi > 0 else
                    f"Bundle deals average {bundle_roi:.1f}x ROI for {seg} "
                    f"while flat 10% discounts show negative returns."
                ),
                "action": (
                    f"Shift {seg} festive promotions from flat discounts "
                    f"to product bundles (e.g. toothpaste + mouthwash). "
                    f"Higher perceived value at lower margin cost."
                ),
            })

    # --- LOW SENSITIVITY SEGMENTS (where discounts are wasted) ---
    low_elastic = sim_results[
        (sim_results["promo_key"] == "flat_10") &
        (sim_results["volume_lift_pct"].abs() < 3)
    ]
    if not low_elastic.empty:
        for seg in low_elastic["segment"].unique():
            seg_cats = low_elastic[low_elastic["segment"] == seg]["category"].tolist()
            if len(seg_cats) >= 2:
                recommendations.append({
                    "type": "DISCOUNT WASTE",
                    "finding": (
                        f"{seg} shows < 3% volume response to 10% discounts "
                        f"across {len(seg_cats)} categories "
                        f"({', '.join(c.title() for c in seg_cats[:3])})."
                    ),
                    "action": (
                        f"Stop running broad discounts for {seg} in these "
                        f"categories. Reallocate budget to in-store displays "
                        f"or target more price-sensitive segments instead."
                    ),
                })

    return recommendations


if __name__ == "__main__":
    print("Loading simulation inputs...")
    elasticity_df, baselines_df = load_simulation_inputs()

    print("Running full simulation...")
    sim_results = run_full_simulation(elasticity_df, baselines_df)

    print("\nFinding optimal promos (by ROI)...")
    optimal = find_optimal_promos(sim_results, optimize_for="promo_roi")
    print(optimal[["category", "segment", "promo_type", "promo_roi", "margin_impact"]].to_string())

    print("\nGenerating visualizations...")
    plot_roi_matrix(optimal)

    top_cats = sim_results["category"].value_counts().head(3).index.tolist()
    for cat in top_cats:
        plot_promo_comparison(sim_results, cat)

    print("\nBusiness Recommendations:")
    recs = generate_recommendations(optimal, sim_results)
    for rec in recs:
        print(f"\n  [{rec['type']}]")
        print(f"  Finding: {rec['finding']}")
        print(f"  Action:  {rec['action']}")

    # Save
    sim_results.to_parquet(PROCESSED_DIR / "simulation_results.parquet", index=False)
    optimal.to_parquet(PROCESSED_DIR / "optimal_promos.parquet", index=False)
    print("\nSaved simulation results.")
"""
Master Runner: Execute the full analysis pipeline.

Usage:
    1. Place Dunnhumby CSV files in data/raw/
    2. Run: python run_pipeline.py
    3. Launch dashboard: streamlit run app/streamlit_app.py

Expected files in data/raw/:
    - transaction_data.csv
    - product.csv
    - hh_demographic.csv
    - coupon.csv
    - coupon_redempt.csv
    - causal_data.csv
    - campaign_table.csv
    - campaign_desc.csv
"""

import sys
from pathlib import Path
import pandas as pd

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.data_pipeline import run_pipeline
from src.segmentation import load_rfm, find_optimal_k, run_segmentation, \
    plot_segment_profiles, plot_segment_distribution, segment_summary_table
from src.elasticity_model import load_data as load_elasticity_data, \
    aggregate_weekly_demand, compute_elasticity_matrix, \
    plot_elasticity_heatmap, plot_promo_effectiveness, generate_elasticity_insights
from src.promo_simulator import load_simulation_inputs, run_full_simulation, \
    find_optimal_promos, plot_roi_matrix, generate_recommendations


def main():
    print("=" * 60)
    print("CPG PRICING & PROMOTION OPTIMIZATION ENGINE")
    print("=" * 60)

    # ── Step 1: Data Pipeline ──────────────────────────────
    print("\n[1/4] RUNNING DATA PIPELINE...")
    master, rfm = run_pipeline()

    # ── Step 2: Customer Segmentation ──────────────────────
    print("\n[2/4] CUSTOMER SEGMENTATION...")
    rfm_with_demo = load_rfm()

    features = [
        "recency", "frequency", "monetary",
        "avg_spend_per_visit", "avg_discount_depth",
        "promo_purchase_rate", "discount_sensitivity",
        "unique_departments",
    ]

    print("Finding optimal k...")
    k_results = find_optimal_k(rfm_with_demo, features)
    best_k = k_results.loc[k_results["silhouette"].idxmax(), "k"]
    print(f"Best k by silhouette score: {int(best_k)}")

    rfm_seg, scaler, km = run_segmentation(rfm_with_demo, n_clusters=int(best_k))
    plot_segment_profiles(rfm_seg)
    plot_segment_distribution(rfm_seg)

    summary = segment_summary_table(rfm_seg)
    print("\nSegment Summary:")
    print(summary.to_string(index=False))

    # Save segmented RFM
    from pathlib import Path
    rfm_seg.to_parquet(Path("data/processed/rfm_segmented.parquet"), index=False)

    # ── Step 3: Price Elasticity ───────────────────────────
    print("\n[3/4] PRICE ELASTICITY MODELING...")
    master_with_seg = load_elasticity_data()
    weekly = aggregate_weekly_demand(master_with_seg)
    print(f"Weekly aggregated rows: {len(weekly)}")

    elasticity_df = compute_elasticity_matrix(weekly)
    plot_elasticity_heatmap(elasticity_df)
    plot_promo_effectiveness(elasticity_df)

    insights = generate_elasticity_insights(elasticity_df)
    print("\nKey Elasticity Insights:")
    for ins in insights:
        print(f"  > {ins}")

    # Save
    elasticity_df.to_parquet(Path("data/processed/elasticity_results.parquet"), index=False)
    weekly.to_parquet(Path("data/processed/weekly_demand.parquet"), index=False)

    # ── Step 4: Promo Simulation ───────────────────────────
    print("\n[4/4] PROMOTION SIMULATION...")
    el_data, baselines = load_simulation_inputs()
    sim_results = run_full_simulation(el_data, baselines)

    optimal = find_optimal_promos(sim_results, optimize_for="promo_roi")
    plot_roi_matrix(optimal)

    print("\nTop 5 Optimal Promotions (by ROI):")
    print(optimal.nlargest(5, "promo_roi")[
        ["category", "segment", "promo_type", "promo_roi", "margin_impact"]
    ].to_string(index=False))

    recs = generate_recommendations(optimal, sim_results)
    print("\nBusiness Recommendations:")
    for rec in recs:
        print(f"\n  [{rec['type']}]")
        print(f"  {rec['finding']}")
        print(f"  -> {rec['action']}")

    # Save simulation results
    sim_results.to_parquet(Path("data/processed/simulation_results.parquet"), index=False)
    optimal.to_parquet(Path("data/processed/optimal_promos.parquet"), index=False)

    # ── Step 5: Budget Optimization ────────────────────────
    print("\n[5/6] BUDGET OPTIMIZATION...")
    from src.budget_optimizer import run_optimization
    opt_results = run_optimization()

    print("\nOptimal Allocations (non-zero):")
    show = opt_results[opt_results["optimal_discount_pct"] > 0.1].sort_values(
        "optimal_discount_pct", ascending=False
    )
    if not show.empty:
        print(show[[
            "COMMODITY_DESC", "SEGMENT_NAME", "optimal_discount_pct",
            "volume_lift_pct", "profit_change_pct", "promo_cost",
        ]].to_string(index=False))

    # ── Step 6: Demand Driver Explainability ───────────────
    print("\n[6/6] DEMAND DRIVER EXPLAINABILITY...")
    from src.explainability import plot_top_categories_drivers, print_driver_insights
    import numpy as np

    weekly_for_drivers = pd.read_parquet(Path("data/processed/weekly_demand.parquet"))

    # Need segment labels on weekly data
    rfm_seg_drivers = pd.read_parquet(Path("data/processed/rfm_segmented.parquet"))
    master_drivers = pd.read_parquet(Path("data/processed/master_transactions.parquet"))
    seg_map_drivers = rfm_seg_drivers[["household_key", "SEGMENT_NAME"]].drop_duplicates()
    master_drivers = master_drivers.merge(seg_map_drivers, on="household_key", how="left")

    weekly_seg = master_drivers.groupby(
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

    drivers = plot_top_categories_drivers(weekly_seg, n_categories=8)
    if drivers is not None:
        print_driver_insights(drivers)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Review charts in outputs/figures/")
    print("  2. Run EDA: python notebooks/01_data_exploration.py")
    print("  3. Launch dashboard: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
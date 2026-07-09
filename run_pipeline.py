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

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.data_pipeline import run_pipeline
from src.segmentation import load_rfm, find_optimal_k, run_segmentation, \
    plot_segment_profiles, plot_segment_distribution, segment_summary_table
from src.elasticity_model import load_data as load_elasticity_data, \
    aggregate_weekly_demand, compute_elasticity_matrix, \
    plot_elasticity_heatmap, plot_promo_effectiveness, generate_elasticity_insights, \
    plot_category_demand_drivers
from src.promo_simulator import load_simulation_inputs, run_full_simulation, \
    find_optimal_promos, plot_roi_matrix, generate_recommendations, \
    optimize_promo_budget, plot_budget_allocation


def main():
    print("=" * 60)
    print("RETAIL PROMOTION INTELLIGENCE PLATFORM")
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

    top_category = weekly.groupby("COMMODITY_DESC")["n_transactions"].sum().idxmax()
    top_segment = (
        weekly[weekly["COMMODITY_DESC"] == top_category]
        .groupby("SEGMENT_NAME")["n_transactions"]
        .sum()
        .idxmax()
    )
    plot_category_demand_drivers(weekly, top_category, top_segment)

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

    total_budget = 50_000
    budget_plan = optimize_promo_budget(sim_results, total_budget=total_budget)
    plot_budget_allocation(budget_plan, total_budget=total_budget)

    print("\nTop 5 Optimal Promotions (by ROI):")
    print(optimal.nlargest(5, "promo_roi")[
        ["category", "segment", "promo_type", "promo_roi", "margin_impact"]
    ].to_string(index=False))

    print(f"\nBudget-Optimized Allocation (${total_budget:,.0f} total):")
    print(budget_plan.head(5)[
        ["category", "segment", "promo_type", "allocation_ratio", "allocated_budget", "expected_objective_value"]
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
    budget_plan.to_parquet(Path("data/processed/budget_allocation.parquet"), index=False)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Review charts in outputs/figures/")
    print("  2. Run EDA: python notebooks/01_data_exploration.py")
    print("  3. Launch dashboard: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
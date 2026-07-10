"""
CPG Pricing & Promotion Optimization Dashboard
Streamlit app for interactive exploration of segmentation,
elasticity, and promo simulation results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.promo_simulator import PROMO_TYPES, simulate_single_promo, DEFAULT_GROSS_MARGIN
from src.budget_optimizer import optimize_budget, plot_optimal_allocation, run_optimization
from src.elasticity_model import plot_category_demand_drivers


PROCESSED_DIR = ROOT_DIR / "data" / "processed"


# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Promotion Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_data():
    budget_path = PROCESSED_DIR / "optimization_results.parquet"
    return {
        "rfm": pd.read_parquet(PROCESSED_DIR / "rfm_segmented.parquet"),
        "elasticity": pd.read_parquet(PROCESSED_DIR / "elasticity_results.parquet"),
        "weekly": pd.read_parquet(PROCESSED_DIR / "weekly_demand.parquet"),
        "simulation": pd.read_parquet(PROCESSED_DIR / "simulation_results.parquet"),
        "optimal": pd.read_parquet(PROCESSED_DIR / "optimal_promos.parquet"),
        "budget_plan": pd.read_parquet(budget_path) if budget_path.exists() else pd.DataFrame(),
    }


data = load_data()


# ── Sidebar ────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Segment Explorer", "Elasticity Map", "Promo Simulator", "Recommendations"],
)


# ═══════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ═══════════════════════════════════════════════════════════
if page == "Overview":
    st.title("Retail Promotion Intelligence Platform")
    st.markdown(
        "Interactive decision-support tool for optimizing pricing, discounts, "
        "and promotional strategies across customer segments."
    )

    rfm = data["rfm"]
    sim = data["simulation"]

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Households", f"{len(rfm):,}")
    col2.metric("Customer Segments", rfm["SEGMENT_NAME"].nunique())
    col3.metric("Categories Analyzed", data["elasticity"]["COMMODITY_DESC"].nunique())
    col4.metric("Scenarios Simulated", f"{len(sim):,}")

    st.divider()

    # Segment overview
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Segment Revenue Share")
        seg_rev = rfm.groupby("SEGMENT_NAME")["monetary"].sum().reset_index()
        seg_rev["pct"] = seg_rev["monetary"] / seg_rev["monetary"].sum() * 100
        fig = px.pie(seg_rev, names="SEGMENT_NAME", values="monetary",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Segment Discount Sensitivity")
        seg_disc = rfm.groupby("SEGMENT_NAME")["discount_sensitivity"].mean().reset_index()
        fig = px.bar(seg_disc, x="SEGMENT_NAME", y="discount_sensitivity",
                     color="discount_sensitivity", color_continuous_scale="Reds",
                     labels={"discount_sensitivity": "Avg Discount Sensitivity"})
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# PAGE 2: SEGMENT EXPLORER
# ═══════════════════════════════════════════════════════════
elif page == "Segment Explorer":
    st.title("Customer Segment Explorer")

    rfm = data["rfm"]
    segments = sorted(rfm["SEGMENT_NAME"].unique())

    selected = st.selectbox("Select Segment", segments)
    seg_data = rfm[rfm["SEGMENT_NAME"] == selected]

    # Segment profile
    st.subheader(f"Profile: {selected}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Households", f"{len(seg_data):,}")
    col2.metric("Avg Spend", f"${seg_data['monetary'].mean():,.0f}")
    col3.metric("Avg Frequency", f"{seg_data['frequency'].mean():.0f} visits")
    col4.metric("Promo Purchase Rate", f"{seg_data['promo_purchase_rate'].mean():.1%}")

    st.divider()

    # Comparison radar
    features = ["frequency", "monetary", "avg_spend_per_visit",
                "promo_purchase_rate", "discount_sensitivity"]
    labels = ["Frequency", "Total Spend", "Basket Value",
              "Promo Rate", "Discount Sensitivity"]

    all_profiles = rfm.groupby("SEGMENT_NAME")[features].median()
    norm = (all_profiles - all_profiles.min()) / (all_profiles.max() - all_profiles.min() + 1e-9)

    fig = go.Figure()
    for seg in segments:
        vals = norm.loc[seg].tolist() + [norm.loc[seg].tolist()[0]]
        opacity = 1.0 if seg == selected else 0.2
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=labels + [labels[0]], name=seg,
            opacity=opacity, fill="toself",
        ))

    fig.update_layout(
        title=f"{selected} vs Other Segments",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Demographics breakdown (if available)
    if "INCOME_DESC" in rfm.columns:
        st.subheader("Demographics")
        col1, col2 = st.columns(2)
        with col1:
            income = seg_data["INCOME_DESC"].value_counts().reset_index()
            income.columns = ["Income", "Count"]
            fig = px.bar(income, x="Income", y="Count", title="Income Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            if "HOUSEHOLD_SIZE_DESC" in rfm.columns:
                hh = seg_data["HOUSEHOLD_SIZE_DESC"].value_counts().reset_index()
                hh.columns = ["HH Size", "Count"]
                fig = px.bar(hh, x="HH Size", y="Count", title="Household Size")
                st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# PAGE 3: ELASTICITY MAP
# ═══════════════════════════════════════════════════════════
elif page == "Elasticity Map":
    st.title("Price Elasticity Map")
    st.markdown(
        "**How to read:** More negative = more price sensitive. "
        "A value of -1.5 means a 10% price drop increases demand by 15%."
    )

    el = data["elasticity"]

    # Filter controls
    min_obs = st.slider("Minimum observations per estimate", 10, 50, 20)
    sig_only = st.checkbox("Show only statistically significant (p < 0.05)", value=True)

    filtered = el[el["n_observations"] >= min_obs]
    if sig_only:
        filtered = filtered[filtered["is_significant"]]

    if filtered.empty:
        st.warning("No results match these filters. Try relaxing them.")
    else:
        # Heatmap
        pivot = filtered.pivot_table(
            index="COMMODITY_DESC", columns="SEGMENT_NAME",
            values="elasticity", aggfunc="mean",
        )

        fig = px.imshow(
            pivot, color_continuous_scale="RdYlGn",
            color_continuous_midpoint=-1, aspect="auto",
            labels=dict(x="Segment", y="Category", color="Elasticity"),
        )
        fig.update_layout(height=max(400, len(pivot) * 35), width=900)
        st.plotly_chart(fig, use_container_width=True)

        # Promo channel effectiveness
        st.subheader("Promotion Channel Effectiveness")
        channels = filtered.groupby("SEGMENT_NAME").agg(
            Discount=("promo_effect", "mean"),
            Display=("display_effect", "mean"),
            Mailer=("mailer_effect", "mean"),
        ).reset_index()

        ch_melt = channels.melt(id_vars="SEGMENT_NAME", var_name="Channel", value_name="Effect")
        fig = px.bar(ch_melt, x="SEGMENT_NAME", y="Effect", color="Channel",
                     barmode="group", title="Which promo channel works best per segment?")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Category Demand Drivers")
        driver_col1, driver_col2 = st.columns(2)
        with driver_col1:
            driver_category = st.selectbox(
                "Category for demand driver view",
                sorted(data["weekly"]["COMMODITY_DESC"].unique()),
                key="driver_category",
            )
        with driver_col2:
            driver_segment = st.selectbox(
                "Segment for demand driver view",
                sorted(data["weekly"]["SEGMENT_NAME"].unique()),
                key="driver_segment",
            )

        driver_result = plot_category_demand_drivers(data["weekly"], driver_category, driver_segment)
        if driver_result:
            driver_fig, _, _ = driver_result
            st.plotly_chart(driver_fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# PAGE 4: PROMO SIMULATOR
# ═══════════════════════════════════════════════════════════
elif page == "Promo Simulator":
    st.title("Promotion Scenario Simulator")
    st.markdown("Compare promotion strategies side by side.")

    el = data["elasticity"]
    weekly = data["weekly"]

    # Baselines
    baselines = weekly.groupby(["COMMODITY_DESC", "SEGMENT_NAME"]).agg(
        base_weekly_quantity=("total_quantity", "median"),
        base_weekly_revenue=("total_revenue", "median"),
        base_avg_price=("avg_unit_price", "median"),
        base_shelf_price=("avg_shelf_unit_price", "median"),
    ).reset_index()

    # Available categories (must have both elasticity and baseline data)
    available = set(el["COMMODITY_DESC"]) & set(baselines["COMMODITY_DESC"])
    categories = sorted(available)
    segments = sorted(el["SEGMENT_NAME"].unique())

    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox("Product Category", categories)
    with col2:
        segment = st.selectbox("Customer Segment", segments)

    st.divider()

    # Simulate all promo types for this combo
    results = []
    for key in PROMO_TYPES:
        r = simulate_single_promo(category, segment, key, el, baselines)
        if r:
            results.append(r)

    if not results:
        st.error("No elasticity estimate available for this category-segment combination.")
    else:
        results_df = pd.DataFrame(results)

        # Comparison table
        st.subheader("Scenario Comparison")
        display_cols = [
            "promo_type", "discount_depth", "volume_lift_pct",
            "incremental_revenue", "margin_impact", "promo_roi",
            "net_incremental_revenue",
        ]
        st.dataframe(
            results_df[display_cols].style.format({
                "discount_depth": "{:.0%}",
                "volume_lift_pct": "{:+.1f}%",
                "incremental_revenue": "${:,.0f}",
                "margin_impact": "${:,.0f}",
                "promo_roi": "{:.2f}x",
                "net_incremental_revenue": "${:,.0f}",
            }).background_gradient(subset=["promo_roi"], cmap="Greens"),
            use_container_width=True,
            hide_index=True,
        )

        # Visual comparison
        col_l, col_r = st.columns(2)
        with col_l:
            fig = px.bar(
                results_df, x="promo_type", y="promo_roi",
                color="promo_roi", color_continuous_scale="Greens",
                title="Promo ROI by Type",
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            fig = px.bar(
                results_df, x="promo_type", y=["incremental_revenue", "margin_impact"],
                barmode="group", title="Revenue Lift vs Margin Impact",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Best recommendation
        best = results_df.loc[results_df["promo_roi"].idxmax()]
        st.success(
            f"**Recommended:** {best['promo_type']} — "
            f"{best['volume_lift_pct']:+.0f}% volume lift, "
            f"ROI of {best['promo_roi']:.1f}x, "
            f"net incremental revenue of ${best['net_incremental_revenue']:,.0f} "
            f"over {best['duration_weeks']} weeks."
        )


# ═══════════════════════════════════════════════════════════
# PAGE 5: RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════
elif page == "Recommendations":
    st.title("Business Recommendations")
    st.markdown(
        "Actionable insights for brand managers and trade marketing teams. "
        "These recommendations are derived from elasticity modeling and "
        "promotion simulation across all category-segment combinations."
    )

    optimal = data["optimal"]
    sim = data["simulation"]

    # Budget allocation optimizer
    st.subheader("Budget Allocation Optimizer")
    promo_budget = st.slider("Total promo budget ($)", 1_000, 50_000, 5_000, step=1_000)

    try:
        budget_result = run_optimization(budget=promo_budget)
        # run_optimization returns a DataFrame directly
        budget_plan = budget_result if isinstance(budget_result, pd.DataFrame) else budget_result[0]
    except Exception as e:
        st.error(f"Optimization error: {e}")
        budget_plan = pd.DataFrame()

    if budget_plan.empty:
        st.info("No budget allocation plan could be generated from the current scenarios.")
    else:
        # Filter to non-zero allocations
        active = budget_plan[budget_plan["optimal_discount_pct"] > 0.1].copy()

        if active.empty:
            st.info("Optimizer found no profitable discount allocations at this budget level.")
        else:
            active = active.sort_values("profit_change_pct", ascending=False)

            budget_metrics = st.columns(4)
            budget_metrics[0].metric("Budget", f"${promo_budget:,.0f}")
            budget_metrics[1].metric("Spent", f"${active['promo_cost'].sum():,.0f}")
            budget_metrics[2].metric(
                "Profit Lift",
                f"${active['profit_change'].sum():,.0f}",
            )
            budget_metrics[3].metric(
                "Top Discount",
                f"{active.iloc[0]['optimal_discount_pct']:.1f}%",
            )

            fig = plot_optimal_allocation(budget_plan)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                active[[
                    "COMMODITY_DESC", "SEGMENT_NAME", "optimal_discount_pct",
                    "volume_lift_pct", "profit_change_pct", "promo_cost",
                ]].rename(columns={
                    "COMMODITY_DESC": "Category",
                    "SEGMENT_NAME": "Segment",
                    "optimal_discount_pct": "Discount %",
                    "volume_lift_pct": "Volume Lift %",
                    "profit_change_pct": "Profit Change %",
                    "promo_cost": "Promo Cost ($)",
                }).style.format({
                    "Discount %": "{:.1f}%",
                    "Volume Lift %": "{:+.1f}%",
                    "Profit Change %": "{:+.1f}%",
                    "Promo Cost ($)": "${:,.0f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

    # Top opportunities
    st.subheader("Highest ROI Promotion Opportunities")
    top_source = optimal[optimal["margin_impact"] > 0].copy()
    if top_source.empty:
        top_source = optimal
    top5 = top_source.nlargest(5, "promo_roi")
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        with st.expander(
            f"#{i}: {row['category']} — {row['segment']} ({row['promo_roi']:.1f}x ROI)",
            expanded=(i <= 3),
        ):
            col1, col2, col3 = st.columns(3)
            col1.metric("Best Promo Type", row["promo_type"])
            col2.metric("Volume Lift", f"{row['volume_lift_pct']:+.0f}%")
            col3.metric("Margin Impact", f"${row['margin_impact']:,.0f}")

            st.markdown(
                f"**Action:** Run **{row['promo_type']}** for "
                f"**{row['category']}** targeting **{row['segment']}** customers. "
                f"Expected ROI of **{row['promo_roi']:.1f}x** over "
                f"{row['duration_weeks']} weeks."
            )

    st.divider()

    # Where to NOT promote
    st.subheader("Margin Risk: Where to Reduce Promotions")
    worst = sim[sim["margin_impact"] < 0].nsmallest(5, "margin_impact")
    if not worst.empty:
        for _, row in worst.iterrows():
            st.error(
                f"**{row['promo_type']}** on **{row['category']}** for "
                f"**{row['segment']}**: destroys **${abs(row['margin_impact']):,.0f}** "
                f"in margin despite {row['volume_lift_pct']:+.0f}% volume lift. "
                f"Recommend discontinuing or reducing depth."
            )
    else:
        st.info("No margin-negative scenarios detected.")

    st.divider()

    # Strategy shifts
    st.subheader("Strategic Shifts")
    for seg in sim["segment"].unique():
        seg_data = sim[sim["segment"] == seg]
        bundle_roi = seg_data[seg_data["promo_key"] == "bundle"]["promo_roi"].mean()
        flat10_roi = seg_data[seg_data["promo_key"] == "flat_10"]["promo_roi"].mean()

        if pd.notna(bundle_roi) and pd.notna(flat10_roi) and bundle_roi > flat10_roi * 1.2:
            st.info(
                f"**{seg}:** Bundle deals yield **{bundle_roi:.1f}x** avg ROI vs "
                f"**{flat10_roi:.1f}x** for flat 10% discounts. "
                f"Shift to combo packs during festive periods."
            )


# ── Footer ─────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.caption(
    "Retail Promotion Intelligence Platform | "
    "Data: Dunnhumby - The Complete Journey"
)
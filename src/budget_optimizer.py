"""
Promotion Budget Optimizer.
Given a fixed marketing budget, finds the optimal allocation of
promotional spend across categories and segments to maximize
total profit (or revenue) subject to business constraints.

Uses scipy.optimize.minimize (SLSQP) for constrained optimization.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = ROOT_DIR / "outputs" / "figures"


def load_optimizer_inputs():
    """Load elasticity and baseline data for optimization."""
    elasticity = pd.read_parquet(PROCESSED_DIR / "elasticity_results.parquet")
    weekly = pd.read_parquet(PROCESSED_DIR / "weekly_demand.parquet")

    # Exclude junk categories
    EXCLUDE = [
        "COUPON/MISC ITEMS", "MISC ITEMS", "MISCELLANEOUS",
        "COUPON", "NO COMMODITY DESCRIPTION",
    ]
    elasticity = elasticity[~elasticity["COMMODITY_DESC"].isin(EXCLUDE)]
    elasticity = elasticity[elasticity["elasticity"] < -0.1].copy()
    weekly = weekly[~weekly["COMMODITY_DESC"].isin(EXCLUDE)]

    # Build optimization input table: one row per category-segment
    # with elasticity + baseline demand
    baselines = weekly.groupby(["COMMODITY_DESC", "SEGMENT_NAME"]).agg(
        base_qty=("total_quantity", "median"),
        base_revenue=("total_revenue", "median"),
        base_price=("avg_unit_price", "median"),
    ).reset_index()

    opt_input = elasticity[["COMMODITY_DESC", "SEGMENT_NAME", "elasticity"]].merge(
        baselines, on=["COMMODITY_DESC", "SEGMENT_NAME"], how="inner"
    )

    # Drop rows with missing or invalid data
    opt_input = opt_input.dropna(subset=["elasticity", "base_qty", "base_price"])
    opt_input = opt_input[opt_input["base_qty"] > 0].copy()
    opt_input = opt_input.reset_index(drop=True)

    return opt_input


def compute_profit(discount_vector, opt_input, gross_margin=0.35, duration_weeks=4):
    """
    Given a vector of discount depths (one per category-segment pair),
    compute total expected profit across the promotion period.

    This is the OBJECTIVE FUNCTION for the optimizer.
    Negative because scipy minimizes (we want to maximize profit).
    """
    total_profit = 0.0

    for i, (_, row) in enumerate(opt_input.iterrows()):
        discount = discount_vector[i]
        elasticity = row["elasticity"]
        base_qty = row["base_qty"]
        base_price = row["base_price"]
        cogs = base_price * (1 - gross_margin)

        # New price after discount
        new_price = base_price * (1 - discount)

        # Demand response (log-log elasticity)
        pct_qty_change = elasticity * (-discount)  # elasticity is negative, discount is positive
        new_qty = base_qty * (1 + pct_qty_change)
        new_qty = max(new_qty, 0)  # can't have negative demand

        # Profit = (price - cogs) * quantity * weeks
        profit = (new_price - cogs) * new_qty * duration_weeks
        total_profit += profit

    return -total_profit  # negative because we minimize


def compute_total_promo_cost(discount_vector, opt_input, duration_weeks=4):
    """Compute total promotional cost (discount given away)."""
    total_cost = 0.0
    for i, (_, row) in enumerate(opt_input.iterrows()):
        discount = discount_vector[i]
        elasticity = row["elasticity"]
        base_qty = row["base_qty"]
        base_price = row["base_price"]

        pct_qty_change = elasticity * (-discount)
        new_qty = base_qty * (1 + pct_qty_change)
        new_qty = max(new_qty, 0)

        cost = discount * base_price * new_qty * duration_weeks
        total_cost += cost

    return total_cost


def optimize_budget(
    opt_input,
    total_budget,
    max_discount=0.25,
    min_margin_pct=0.10,
    gross_margin=0.35,
    duration_weeks=4,
):
    """
    Find the optimal discount depth for each category-segment pair
    that maximizes total profit subject to:

    1. Total promo spend <= total_budget
    2. Per-item discount between 0% and max_discount
    3. Per-item margin stays above min_margin_pct of base price

    Returns:
        results_df: DataFrame with optimal discount per category-segment
        opt_result: scipy OptimizeResult object
    """
    n = len(opt_input)

    # Initial guess: uniform small discount
    x0 = np.full(n, 0.05)

    # Bounds: each discount between 0 and max_discount
    bounds = [(0.0, max_discount)] * n

    # Constraint 1: total promo cost <= budget
    def budget_constraint(x):
        return total_budget - compute_total_promo_cost(x, opt_input, duration_weeks)

    # Constraint 2: margin floor per item
    # (1 - discount) * base_price - cogs >= min_margin_pct * base_price
    # => discount <= 1 - (cogs + min_margin_pct * base_price) / base_price
    # => discount <= gross_margin - min_margin_pct
    margin_cap = gross_margin - min_margin_pct
    bounds = [(0.0, min(max_discount, margin_cap))] * n

    constraints = [
        {"type": "ineq", "fun": budget_constraint},
    ]

    print(f"Optimizing {n} category-segment pairs...")
    print(f"  Budget: ${total_budget:,.0f}")
    print(f"  Max discount: {max_discount:.0%}")
    print(f"  Min margin: {min_margin_pct:.0%}")

    result = minimize(
        compute_profit,
        x0,
        args=(opt_input, gross_margin, duration_weeks),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-8},
    )

    if result.success:
        print(f"  Optimization converged.")
    else:
        print(f"  Warning: {result.message}")

    # Build results table
    results = opt_input.copy()
    results["optimal_discount"] = result.x
    results["optimal_discount_pct"] = (result.x * 100).round(1)

    # Compute outcomes at optimal discount
    outcomes = []
    for i, (_, row) in enumerate(results.iterrows()):
        d = result.x[i]
        e = row["elasticity"]
        bq = row["base_qty"]
        bp = row["base_price"]
        cogs = bp * (1 - gross_margin)

        new_price = bp * (1 - d)
        pct_change = e * (-d)
        new_qty = bq * (1 + pct_change)

        base_profit = (bp - cogs) * bq * duration_weeks
        new_profit = (new_price - cogs) * new_qty * duration_weeks
        promo_cost = d * bp * new_qty * duration_weeks
        volume_lift = pct_change * 100

        outcomes.append({
            "new_qty": round(new_qty),
            "volume_lift_pct": round(volume_lift, 1),
            "base_profit": round(base_profit, 2),
            "optimized_profit": round(new_profit, 2),
            "profit_change": round(new_profit - base_profit, 2),
            "profit_change_pct": round((new_profit - base_profit) / base_profit * 100, 1) if base_profit > 0 else 0,
            "promo_cost": round(promo_cost, 2),
        })

    outcomes_df = pd.DataFrame(outcomes)
    results = pd.concat([results.reset_index(drop=True), outcomes_df], axis=1)

    # Summary
    total_base_profit = results["base_profit"].sum()
    total_opt_profit = results["optimized_profit"].sum()
    total_cost = results["promo_cost"].sum()

    print(f"\n  Baseline profit (no promo):   ${total_base_profit:,.0f}")
    print(f"  Optimized profit:             ${total_opt_profit:,.0f}")
    print(f"  Profit improvement:           ${total_opt_profit - total_base_profit:,.0f} "
          f"({(total_opt_profit - total_base_profit) / total_base_profit * 100:+.1f}%)")
    print(f"  Total promo spend:            ${total_cost:,.0f} / ${total_budget:,.0f} budget")

    return results, result


def plot_optimal_allocation(results):
    """Visualize the optimal budget allocation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Sort by optimal discount
    res = results.sort_values("optimal_discount_pct", ascending=False).copy()
    res["label"] = res["COMMODITY_DESC"].str.title() + " — " + res["SEGMENT_NAME"]

    # Only show allocations > 0
    res = res[res["optimal_discount_pct"] > 0.1]

    if res.empty:
        print("No non-zero allocations to plot.")
        return None

    # Chart 1: Optimal discount by category-segment
    fig = px.bar(
        res, x="label", y="optimal_discount_pct",
        color="SEGMENT_NAME",
        title="Optimal Discount Depth by Category & Segment<br><sub>Constrained by budget, max discount, and minimum margin</sub>",
        labels={"optimal_discount_pct": "Optimal Discount (%)", "label": ""},
        text="optimal_discount_pct",
        color_discrete_sequence=["#2196F3", "#FF9800", "#4CAF50"],
    )
    fig.update_layout(
        height=500, width=1000,
        xaxis_tickangle=-40,
        showlegend=True,
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.write_html(str(OUTPUT_DIR / "optimal_allocation.html"))
    try:
        fig.write_image(str(OUTPUT_DIR / "optimal_allocation.png"), scale=2)
    except Exception:
        pass
    print("Saved optimal allocation chart.")

    # Chart 2: Profit impact waterfall
    fig2 = go.Figure(go.Waterfall(
        x=["Baseline Profit", "Promo Volume Lift", "Discount Cost", "Optimized Profit"],
        y=[
            results["base_profit"].sum(),
            results["optimized_profit"].sum() - results["base_profit"].sum() + results["promo_cost"].sum(),
            -results["promo_cost"].sum(),
            results["optimized_profit"].sum(),
        ],
        measure=["absolute", "relative", "relative", "total"],
        text=[
            f"${results['base_profit'].sum():,.0f}",
            f"+${results['optimized_profit'].sum() - results['base_profit'].sum() + results['promo_cost'].sum():,.0f}",
            f"-${results['promo_cost'].sum():,.0f}",
            f"${results['optimized_profit'].sum():,.0f}",
        ],
        connector={"line": {"color": "#ccc"}},
        increasing={"marker": {"color": "#4CAF50"}},
        decreasing={"marker": {"color": "#FF5722"}},
        totals={"marker": {"color": "#2196F3"}},
    ))
    fig2.update_layout(
        title="Profit Impact Waterfall<br><sub>How promotional spend translates to profit change</sub>",
        height=450, width=800,
        yaxis_title="Profit ($)",
    )
    fig2.write_html(str(OUTPUT_DIR / "profit_waterfall.html"))
    try:
        fig2.write_image(str(OUTPUT_DIR / "profit_waterfall.png"), scale=2)
    except Exception:
        pass
    print("Saved profit waterfall chart.")

    # Chart 3: Budget allocation pie
    res_cost = res[res["promo_cost"] > 0].copy()
    if not res_cost.empty:
        fig3 = px.pie(
            res_cost, names="label", values="promo_cost",
            title="Promotional Budget Allocation<br><sub>Where the optimizer chose to spend</sub>",
        )
        fig3.update_layout(height=500, width=700)
        fig3.write_html(str(OUTPUT_DIR / "budget_allocation.html"))
        try:
            fig3.write_image(str(OUTPUT_DIR / "budget_allocation.png"), scale=2)
        except Exception:
            pass
        print("Saved budget allocation chart.")

    return fig


def run_optimization(budget=None):
    """Run the full optimization pipeline."""
    opt_input = load_optimizer_inputs()
    print(f"Loaded {len(opt_input)} category-segment pairs for optimization.")

    if budget is None:
        # Default: 10% of total baseline revenue as promo budget
        total_rev = opt_input["base_revenue"].sum() * 4  # 4-week period
        budget = total_rev * 0.10
        print(f"Auto-set budget to 10% of baseline revenue: ${budget:,.0f}")

    results, opt_result = optimize_budget(
        opt_input,
        total_budget=budget,
        max_discount=0.25,
        min_margin_pct=0.10,
    )

    plot_optimal_allocation(results)

    # Save
    results.to_parquet(PROCESSED_DIR / "optimization_results.parquet", index=False)
    print("Saved optimization results.")

    return results


if __name__ == "__main__":
    results = run_optimization()

    print("\nOptimal Allocations (non-zero):")
    show = results[results["optimal_discount_pct"] > 0.1].sort_values(
        "optimal_discount_pct", ascending=False
    )
    print(show[[
        "COMMODITY_DESC", "SEGMENT_NAME", "optimal_discount_pct",
        "volume_lift_pct", "profit_change_pct", "promo_cost",
    ]].to_string(index=False))
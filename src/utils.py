"""Shared utility functions."""

import pandas as pd
import numpy as np
from pathlib import Path


def format_currency(val):
    """Format number as currency string."""
    if abs(val) >= 1_000_000:
        return f"${val / 1_000_000:.1f}M"
    elif abs(val) >= 1_000:
        return f"${val / 1_000:.1f}K"
    return f"${val:.2f}"


def format_pct(val):
    """Format number as percentage string."""
    return f"{val:.1f}%"


def safe_divide(a, b, default=0):
    """Divide a by b, returning default if b is zero."""
    return a / b if b != 0 else default


def get_top_categories(master_df, n=10):
    """Get top n product categories by transaction volume."""
    return (
        master_df.groupby("COMMODITY_DESC")["BASKET_ID"]
        .nunique()
        .nlargest(n)
        .index.tolist()
    )


def load_all_results():
    """Load all processed results for the Streamlit app."""
    p = Path("data/processed")
    return {
        "master": pd.read_parquet(p / "master_transactions.parquet"),
        "rfm": pd.read_parquet(p / "rfm_segmented.parquet"),
        "elasticity": pd.read_parquet(p / "elasticity_results.parquet"),
        "weekly": pd.read_parquet(p / "weekly_demand.parquet"),
        "simulation": pd.read_parquet(p / "simulation_results.parquet"),
        "optimal": pd.read_parquet(p / "optimal_promos.parquet"),
    }

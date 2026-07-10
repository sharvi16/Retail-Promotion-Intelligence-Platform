"""
Customer Segmentation using RFM Analysis + K-Means Clustering.
Segments households into actionable groups for pricing strategy.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs/figures")


def load_rfm():
    """Load RFM table and demographics."""
    rfm = pd.read_parquet(PROCESSED_DIR / "rfm_table.parquet")
    demo = pd.read_parquet(PROCESSED_DIR / "demographics.parquet")
    return rfm.merge(demo, on="household_key", how="left")


def find_optimal_k(rfm, features, k_range=range(2, 8)):
    """Find optimal number of clusters using silhouette score."""
    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[features].fillna(0))

    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        inertia = km.inertia_
        results.append({"k": k, "silhouette": sil, "inertia": inertia})
        print(f"  k={k}: silhouette={sil:.4f}, inertia={inertia:.0f}")

    return pd.DataFrame(results)


def run_segmentation(rfm, n_clusters=4):
    """Run K-Means segmentation on RFM features."""
    features = [
        "recency",
        "frequency",
        "monetary",
        "avg_spend_per_visit",
        "avg_discount_depth",
        "promo_purchase_rate",
        "discount_sensitivity",
        "unique_departments",
    ]

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[features].fillna(0))

    # Fit K-Means
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["CLUSTER"] = km.fit_predict(X)

    sil = silhouette_score(X, rfm["CLUSTER"])
    print(f"\nSegmentation complete: {n_clusters} clusters, silhouette={sil:.4f}")

    # Merge tiny clusters (< 3% of households) into nearest large cluster
    rfm, km = merge_tiny_clusters(rfm, X, km, min_pct=0.03)

    # Name segments based on cluster profiles
    rfm = name_segments(rfm)

    return rfm, scaler, km


def merge_tiny_clusters(rfm, X, km, min_pct=0.03):
    """
    Merge clusters smaller than min_pct of total into the nearest
    large cluster by centroid distance. Prevents noisy micro-segments.
    """
    counts = rfm["CLUSTER"].value_counts(normalize=True)
    tiny = counts[counts < min_pct].index.tolist()

    if not tiny:
        return rfm, km

    large = counts[counts >= min_pct].index.tolist()
    centroids = km.cluster_centers_

    for small_c in tiny:
        # Find nearest large cluster by centroid distance
        dists = {
            lg: np.linalg.norm(centroids[small_c] - centroids[lg])
            for lg in large
        }
        nearest = min(dists, key=dists.get)
        rfm.loc[rfm["CLUSTER"] == small_c, "CLUSTER"] = nearest
        print(f"  Merged tiny cluster {small_c} ({counts[small_c]:.1%}) into cluster {nearest}")

    # Re-number clusters sequentially
    unique_clusters = sorted(rfm["CLUSTER"].unique())
    remap = {old: new for new, old in enumerate(unique_clusters)}
    rfm["CLUSTER"] = rfm["CLUSTER"].map(remap)

    return rfm, km


def name_segments(rfm):
    """Assign business-meaningful names to clusters based on behavior."""
    # Compute cluster-level medians
    profiles = rfm.groupby("CLUSTER").agg(
        med_recency=("recency", "median"),
        med_frequency=("frequency", "median"),
        med_monetary=("monetary", "median"),
        med_discount_sensitivity=("discount_sensitivity", "median"),
        med_promo_rate=("promo_purchase_rate", "median"),
        count=("household_key", "count"),
    ).reset_index()

    print("\nCluster Profiles:")
    print(profiles.to_string(index=False))

    # Scoring-based naming
    # Sort clusters by monetary (descending) and assign ranked names
    profiles = profiles.sort_values("med_monetary", ascending=False).reset_index(drop=True)

    name_candidates = []
    for _, row in profiles.iterrows():
        high_spend = row["med_monetary"] > profiles["med_monetary"].median()
        high_promo = row["med_promo_rate"] > profiles["med_promo_rate"].median()
        high_freq = row["med_frequency"] > profiles["med_frequency"].median()
        high_recency = row["med_recency"] > profiles["med_recency"].median()

        if high_spend and not high_promo:
            name = "Premium Loyalists"
        elif high_spend and high_promo:
            name = "Deal-Seeking Big Spenders"
        elif not high_spend and high_promo:
            name = "Budget Deal Hunters"
        elif not high_spend and high_recency:
            name = "At-Risk / Lapsing"
        elif high_freq and not high_spend:
            name = "Frequent Low Spenders"
        else:
            name = "Occasional Buyers"

        name_candidates.append({"CLUSTER": row["CLUSTER"], "SEGMENT_NAME": name})

    # Handle duplicate names by appending cluster number
    names_df = pd.DataFrame(name_candidates)
    seen = {}
    for idx, row in names_df.iterrows():
        name = row["SEGMENT_NAME"]
        if name in seen:
            names_df.at[idx, "SEGMENT_NAME"] = f"{name} (B)"
        seen[name] = True

    rfm = rfm.merge(names_df, on="CLUSTER", how="left")
    print("\nSegment Names:")
    for _, row in names_df.iterrows():
        print(f"  Cluster {int(row['CLUSTER'])}: {row['SEGMENT_NAME']}")

    return rfm


def plot_segment_profiles(rfm):
    """Create radar chart of segment profiles."""
    features = [
        "frequency", "monetary", "avg_spend_per_visit",
        "promo_purchase_rate", "discount_sensitivity",
    ]
    labels = [
        "Purchase Frequency", "Total Spend", "Avg Basket Value",
        "Promo Purchase Rate", "Discount Sensitivity",
    ]

    profiles = rfm.groupby("SEGMENT_NAME")[features].median()

    # Use RANK-BASED normalization instead of min-max.
    # Min-max collapses small segments to near-zero when one segment
    # dominates (e.g. Premium Loyalists with 10x the spend).
    # Rank-based gives each segment a proportional position (0 to 1)
    # regardless of absolute magnitude, making the radar readable.
    profiles_norm = profiles.rank(pct=True)

    fig = go.Figure()
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0", "#00BCD4"]

    for i, (segment, row) in enumerate(profiles_norm.iterrows()):
        values = row.tolist() + [row.tolist()[0]]  # close the polygon
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels + [labels[0]],
            name=segment,
            line=dict(color=colors[i % len(colors)], width=2),
            fill="toself",
            opacity=0.4,
        ))

    fig.update_layout(
        title="Customer Segment Profiles<br><sub>Rank-normalized: shows relative position across segments, not absolute values</sub>",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1.05])),
        showlegend=True,
        width=850,
        height=650,
        legend=dict(font=dict(size=12)),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(OUTPUT_DIR / "segment_profiles.html"))

    try:
        fig.write_image(str(OUTPUT_DIR / "segment_profiles.png"), scale=2)
    except Exception:
        pass

    print("Saved segment profile charts.")

    return fig


def plot_segment_distribution(rfm):
    """Plot segment size and revenue contribution."""
    seg_stats = rfm.groupby("SEGMENT_NAME").agg(
        households=("household_key", "count"),
        total_revenue=("monetary", "sum"),
    ).reset_index()

    seg_stats["revenue_pct"] = seg_stats["total_revenue"] / seg_stats["total_revenue"].sum() * 100
    seg_stats["household_pct"] = seg_stats["households"] / seg_stats["households"].sum() * 100

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("% of Households", "% of Revenue"),
        specs=[[{"type": "pie"}, {"type": "pie"}]],
    )

    colors = px.colors.qualitative.Set2
    fig.add_trace(
        go.Pie(labels=seg_stats["SEGMENT_NAME"], values=seg_stats["household_pct"],
               marker=dict(colors=colors), textinfo="label+percent"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Pie(labels=seg_stats["SEGMENT_NAME"], values=seg_stats["revenue_pct"],
               marker=dict(colors=colors), textinfo="label+percent"),
        row=1, col=2,
    )

    fig.update_layout(title="Segment: Household Share vs Revenue Share", width=1000, height=500)

    fig.write_html(str(OUTPUT_DIR / "segment_distribution.html"))
    fig.write_image(str(OUTPUT_DIR / "segment_distribution.png"), scale=2)
    print("Saved segment distribution charts.")

    return fig


def segment_summary_table(rfm):
    """Generate a clean summary table for each segment."""
    summary = rfm.groupby("SEGMENT_NAME").agg(
        households=("household_key", "count"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
        avg_basket=("avg_spend_per_visit", "mean"),
        avg_discount_depth=("avg_discount_depth", "mean"),
        promo_rate=("promo_purchase_rate", "mean"),
        discount_sensitivity=("discount_sensitivity", "mean"),
    ).round(2).reset_index()

    return summary


if __name__ == "__main__":
    print("Loading RFM data...")
    rfm = load_rfm()

    print("\nFinding optimal k...")
    k_results = find_optimal_k(rfm, [
        "recency", "frequency", "monetary",
        "avg_spend_per_visit", "avg_discount_depth",
        "promo_purchase_rate", "discount_sensitivity",
        "unique_departments",
    ])

    print("\nRunning segmentation with k=4...")
    rfm, scaler, km = run_segmentation(rfm, n_clusters=4)

    print("\nGenerating visualizations...")
    plot_segment_profiles(rfm)
    plot_segment_distribution(rfm)

    summary = segment_summary_table(rfm)
    print("\nSegment Summary:")
    print(summary.to_string(index=False))

    # Save segmented RFM
    rfm.to_parquet(PROCESSED_DIR / "rfm_segmented.parquet", index=False)
    print("\nSaved segmented RFM to data/processed/rfm_segmented.parquet")
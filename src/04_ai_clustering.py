"""
04_ai_clustering.py — K-Means clustering of districts by accessibility
======================================================================

This script is the "AI" core of the project.  It takes the normalised
accessibility metrics from step 03 and uses unsupervised machine
learning to group Hong Kong's 18 districts into three categories:

High accessibility
Medium accessibility
Low accessibility   (potential "transport deserts")

Why K-Means?
-----------
K-Means is a solid baseline for spatial data:
-   It's interpretable — you can explain the clusters by looking at
    their centroids (average metric values).
-   It works well with small datasets (we have only 18 districts, so
    complex deep-learning methods would be overkill).
-   It aligns with the 3-tier classification that urban planners
    already use.

Model evaluation
----------------
We run two diagnostic tests:
1.  **Elbow method** — plots inertia vs. k to justify k = 3.
2.  **Silhouette analysis** — measures how well-separated the clusters
    are (values close to 1 are great, near 0 means overlapping).

Output
------
    output/clustered_districts.csv
    output/elbow_plot.png
    output/silhouette_plot.png

Author:  CCAI-9012 Group E
Date:    March 2026
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# The features we feed into the clustering model.
# These were computed and normalised in step 03.
FEATURE_COLS = [
    "norm_stops_per_km2",
    "norm_routes_per_km2",
    "norm_stops_per_10k",
    "norm_walk_inv",
    "norm_operator_div",
    "norm_ramp_cov",
    "norm_terrain_inv",
]

# Labels we assign to each cluster, ordered from best to worst
# accessibility (we'll sort clusters by their centroid scores).
CLUSTER_LABELS = ["High Accessibility", "Medium Accessibility",
                  "Low Accessibility"]


# ===================================================================
# Clustering
# ===================================================================

def run_kmeans(X: np.ndarray, k: int = 3,
               random_state: int = 42) -> KMeans:
    """
    Fit a K-Means model with k clusters.

    We fix the random_state so results are reproducible — important
    for academic work where reviewers need to verify your numbers.
    """
    model = KMeans(
        n_clusters=k,
        init="k-means++",      # smart initialisation (better than random)
        n_init=10,              # run 10 times, keep best
        max_iter=300,
        random_state=random_state,
    )
    model.fit(X)
    return model


def assign_ordered_labels(df: pd.DataFrame, labels: np.ndarray,
                          X_scaled: np.ndarray) -> pd.Series:
    """
    Sort cluster IDs so that cluster 0 = highest average accessibility,
    cluster 2 = lowest.

    K-Means assigns arbitrary cluster numbers.  We re-order them so
    the labels "High / Medium / Low" are semantically correct.
    """
    # Compute mean accessibility score per cluster
    cluster_means = {}
    for cid in np.unique(labels):
        mask = labels == cid
        cluster_means[cid] = df.loc[mask, "accessibility_score"].mean()

    # Sort clusters by descending mean score
    sorted_ids = sorted(cluster_means, key=cluster_means.get, reverse=True)

    # Map: original cluster id → ordered rank
    rank_map = {cid: rank for rank, cid in enumerate(sorted_ids)}
    ordered = np.array([rank_map[c] for c in labels])

    # Convert to descriptive string labels
    return pd.Series([CLUSTER_LABELS[r] for r in ordered],
                     index=df.index, name="cluster_label")


# ===================================================================
# Elbow method
# ===================================================================

def elbow_plot(X: np.ndarray, max_k: int = 8) -> None:
    """
    Plot inertia (within-cluster sum of squares) vs. number of
    clusters to find the "elbow" — the point after which adding
    more clusters gives diminishing returns.

    For our data, we expect the elbow at k = 3.
    """
    inertias = []
    ks = range(2, max_k + 1)

    for k in ks:
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        model.fit(X)
        inertias.append(model.inertia_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, inertias, "o-", color="#2563eb", linewidth=2,
            markersize=8, markerfacecolor="#1d4ed8")
    ax.axvline(x=3, color="#ef4444", linestyle="--", alpha=0.7,
               label="Chosen k = 3")
    ax.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax.set_ylabel("Inertia (WCSS)", fontsize=12)
    ax.set_title("Elbow Method — Selecting Optimal k", fontsize=14,
                 fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    path = os.path.join(OUTPUT_DIR, "elbow_plot.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Elbow plot saved: {path}")


# ===================================================================
# Silhouette analysis
# ===================================================================

def silhouette_plot(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Create a silhouette plot — each horizontal bar represents one
    district, and its width shows how well it fits its cluster.

    Returns the mean silhouette score.
    """
    n_clusters = len(np.unique(labels))
    sil_avg = silhouette_score(X, labels)
    sil_vals = silhouette_samples(X, labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    y_lower = ("10")
    y_lower = 10

    colours = ["#22c55e", "#f59e0b", "#ef4444"]  # green, amber, red
    for i in range(n_clusters):
        cluster_sil = np.sort(sil_vals[labels == i])
        size = len(cluster_sil)
        y_upper = y_lower + size

        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
                         facecolor=colours[i % len(colours)], alpha=0.7,
                         edgecolor="none")
        ax.text(-0.05, y_lower + 0.5 * size,
                CLUSTER_LABELS[i], fontsize=9, va="center")
        y_lower = y_upper + 10

    ax.axvline(x=sil_avg, color="#1e293b", linestyle="--", linewidth=1.5,
               label=f"Mean = {sil_avg:.3f}")
    ax.set_xlabel("Silhouette Coefficient", fontsize=12)
    ax.set_ylabel("Districts (sorted within cluster)", fontsize=12)
    ax.set_title("Silhouette Analysis for K-Means (k = 3)", fontsize=14,
                 fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_yticks([])

    path = os.path.join(OUTPUT_DIR, "silhouette_plot.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Silhouette plot saved: {path}")

    return sil_avg


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    print("=" * 60)
    print("04_ai_clustering  ·  Grouping districts with K-Means")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    scores_path = os.path.join(OUTPUT_DIR, "accessibility_scores.csv")
    df = pd.read_csv(scores_path)
    print(f"\nLoaded {len(df)} districts")

    # Check that all feature columns exist
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"✘ Missing columns: {missing}")
        print("  Run 03_compute_accessibility.py first.")
        return

    # ------------------------------------------------------------------
    # Standardise features
    # ------------------------------------------------------------------
    print("\n[1/3]  Standardising features …")
    X = df[FEATURE_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("       Mean after scaling:", np.round(X_scaled.mean(axis=0), 4))
    print("       Std  after scaling:", np.round(X_scaled.std(axis=0), 4))

    # ------------------------------------------------------------------
    # Elbow method
    # ------------------------------------------------------------------
    print("\n[2/3]  Running elbow method …")
    elbow_plot(X_scaled)

    # ------------------------------------------------------------------
    # Fit K-Means (k = 3)
    # ------------------------------------------------------------------
    print("\n[3/3]  Fitting K-Means (k = 3) …")
    model = run_kmeans(X_scaled, k=3)
    raw_labels = model.labels_

    # Re-order labels so High=0, Medium=1, Low=2
    df["cluster_label"] = assign_ordered_labels(df, raw_labels, X_scaled)
    df["cluster_id"]    = df["cluster_label"].map({
        "High Accessibility": 0,
        "Medium Accessibility": 1,
        "Low Accessibility": 2,
    })

    # Silhouette analysis
    ordered_ids = df["cluster_id"].values
    sil_avg = silhouette_plot(X_scaled, ordered_ids)

    # ------------------------------------------------------------------
    # Print cluster summary
    # ------------------------------------------------------------------
    print(f"\n  Mean silhouette score: {sil_avg:.3f}")
    print(f"\n{'District':<25s} {'Cluster':<25s} {'Score':>8s}")
    print("-" * 60)
    for _, row in df.sort_values("cluster_id").iterrows():
        print(f"{row['district']:<25s} "
              f"{row['cluster_label']:<25s} "
              f"{row['accessibility_score']:>8.3f}")

    # Cluster centroids in original feature space
    print("\n  Cluster centroids (normalised features):")
    for label in CLUSTER_LABELS:
        mask = df["cluster_label"] == label
        means = df.loc[mask, FEATURE_COLS].mean()
        districts = df.loc[mask, "district"].tolist()
        print(f"\n  {label}:")
        for col in FEATURE_COLS:
            print(f"    {col:30s} = {means[col]:.3f}")
        print(f"    Districts: {', '.join(districts)}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path = os.path.join(OUTPUT_DIR, "clustered_districts.csv")
    df.to_csv(out_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"Clustered data saved to: {os.path.abspath(out_path)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

"""
05_visualise_results.py — Generate maps and charts
===================================================

The final visualisation step.  This script creates four outputs:

1.  **Interactive choropleth map** (Folium → HTML) — colour-coded by
    accessibility cluster, with pop-ups showing district metrics.
2.  **Bar chart** — district accessibility scores, colour-coded.
3.  **Radar chart** — average metric profile per cluster.
4.  **Summary report CSV** — a tidy table of everything.

These are the deliverables you'd show to a professor or policymaker.

Output
------
    output/accessibility_map.html
    output/district_scores_bar.png
    output/cluster_profiles_radar.png
    output/summary_report.csv

Author:  CCAI-9012 Group E
Date:    March 2026
"""

import json
import os
import warnings

import folium
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR    = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Colour palette — matches the cluster traffic-light colour scheme
CLUSTER_COLOURS = {
    "High Accessibility":   "#22c55e",   # green
    "Medium Accessibility": "#f59e0b",   # amber
    "Low Accessibility":    "#ef4444",   # red
}

MICRO_GRID_PATH = os.path.join(OUTPUT_DIR, "micro_accessibility_grid.csv")
TOPOGRAPHY_PATH = os.path.join(RAW_DIR, "hk_topography_points.csv")


# ===================================================================
# 1.  Choropleth Map
# ===================================================================

def create_choropleth(df: pd.DataFrame,
                      districts_path: str) -> None:
    """
    Build an interactive Folium map of Hong Kong coloured by
    accessibility cluster.

    Each district polygon is filled with a traffic-light colour
    (green / amber / red) and shows a pop-up when clicked with
    detailed metrics.
    """
    print("  Creating choropleth map …")

    # Load district boundaries
    districts = gpd.read_file(districts_path)

    # Find the English name column
    name_col = None
    for candidate in ["ENAME", "NAME_EN", "name_en", "DCNAME_EN",
                      "ename", "Name", "name", "DISTRICT", "District"]:
        if candidate in districts.columns:
            name_col = candidate
            break
    if name_col:
        districts = districts.rename(columns={name_col: "district"})
    else:
        raise KeyError(
            "Could not find a district name column in the districts GeoJSON. "
            f"Available columns: {districts.columns.tolist()}"
        )

    # Normalise district names for matching
    def norm(s):
        return s.strip().lower().replace("&", "and").replace("  ", " ")

    districts["_key"] = districts["district"].apply(norm)
    df_copy = df.copy()
    df_copy["_key"] = df_copy["district"].apply(norm)

    # Merge cluster info onto the GeoDataFrame
    districts = districts.merge(
        df_copy[["_key", "cluster_label", "accessibility_score",
                 "total_stops", "avg_walking_dist_m", "pct_within_400m",
                 "stops_per_km2", "routes_per_km2", "stops_per_10k",
                 "ramp_coverage_pct", "terrain_ruggedness",
                 "operator_diversity", "population", "area_km2"]],
        on="_key", how="left"
    )

    # Centre the map on Hong Kong
    hk_centre = [22.35, 114.15]
    m = folium.Map(location=hk_centre, zoom_start=11,
                   tiles="CartoDB positron")

    # Style function — fills each polygon with the cluster colour
    def style_fn(feature):
        label = feature["properties"].get("cluster_label", "")
        colour = CLUSTER_COLOURS.get(label, "#9ca3af")
        return {
            "fillColor":   colour,
            "color":       "#1e293b",
            "weight":      1.5,
            "fillOpacity": 0.55,
        }

    # Highlight on hover
    def highlight_fn(feature):
        return {
            "weight":      3,
            "fillOpacity": 0.8,
        }

    # Build the GeoJSON layer with pop-ups
    geojson_data = json.loads(districts.to_json())
    folium.GeoJson(
        geojson_data,
        name="Districts",
        style_function=style_fn,
        highlight_function=highlight_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=["district", "cluster_label", "accessibility_score",
                    "total_stops", "avg_walking_dist_m"],
            aliases=["District", "Category", "Score",
                     "Total Stops", "Avg Walk (m)"],
            style="font-size: 12px;",
        ),
        popup=folium.GeoJsonPopup(
            fields=["district", "cluster_label", "accessibility_score",
                    "total_stops", "population", "area_km2",
                    "stops_per_km2", "routes_per_km2",
                    "stops_per_10k", "avg_walking_dist_m",
                    "pct_within_400m", "ramp_coverage_pct",
                    "terrain_ruggedness", "operator_diversity"],
            aliases=["District", "Category", "Accessibility Score",
                     "Total Stops", "Population", "Area (km²)",
                     "Stops / km²", "Routes / km²",
                     "Stops / 10k People", "Avg Walk (m)",
                     "% Within 400 m", "Ramp Coverage (%)",
                     "Terrain Ruggedness", "Operator Diversity"],
        ),
    ).add_to(m)

    # Legend (manual HTML)
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                background: white; padding: 15px 20px; border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                font-family: Arial, sans-serif; font-size: 13px;">
        <b style="font-size: 14px;">Transport Accessibility</b><br><br>
        <i style="background:#22c55e;width:16px;height:16px;
           display:inline-block;border-radius:3px;margin-right:6px;"></i>
        High Accessibility<br>
        <i style="background:#f59e0b;width:16px;height:16px;
           display:inline-block;border-radius:3px;margin-right:6px;"></i>
        Medium Accessibility<br>
        <i style="background:#ef4444;width:16px;height:16px;
           display:inline-block;border-radius:3px;margin-right:6px;"></i>
        Low Accessibility<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Title
    title_html = """
    <div style="position: fixed; top: 15px; left: 50%; transform: translateX(-50%);
                z-index: 1000; background: rgba(30,41,59,0.9); color: white;
                padding: 10px 24px; border-radius: 8px;
                font-family: Arial, sans-serif; font-size: 16px;
                font-weight: bold; box-shadow: 0 2px 8px rgba(0,0,0,0.3);">
        AI Detection of Urban Transport Inequality — Hong Kong
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    path = os.path.join(OUTPUT_DIR, "accessibility_map.html")
    m.save(path)
    print(f"  ✓ Map saved: {path}")


# ===================================================================
# 2.  Bar Chart
# ===================================================================

def create_bar_chart(df: pd.DataFrame) -> None:
    """
    Horizontal bar chart of accessibility scores, colour-coded by
    cluster assignment.
    """
    print("  Creating bar chart …")

    df_sorted = df.sort_values("accessibility_score", ascending=True)
    colours = [CLUSTER_COLOURS.get(lbl, "#9ca3af")
               for lbl in df_sorted["cluster_label"]]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(df_sorted["district"], df_sorted["accessibility_score"],
                   color=colours, edgecolor="#1e293b", linewidth=0.5)

    ax.set_xlabel("Composite Accessibility Score", fontsize=12)
    ax.set_title("Transport Accessibility by District",
                 fontsize=15, fontweight="bold", pad=15)
    ax.grid(axis="x", alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, df_sorted["accessibility_score"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9, color="#374151")

    # Custom legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=c, label=l)
                      for l, c in CLUSTER_COLOURS.items()]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=10)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "district_scores_bar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Bar chart saved: {path}")


# ===================================================================
# 3.  Radar Chart
# ===================================================================

def create_radar_chart(df: pd.DataFrame) -> None:
    """
    Radar (spider) chart showing the average metric profile for each
    cluster.

    This helps explain *why* certain districts are classified as they
    are — e.g. 'Low Accessibility' districts might have decent per-
    capita stop counts but very high walking distances.
    """
    print("  Creating radar chart …")

    metrics = ["norm_stops_per_km2", "norm_routes_per_km2",
               "norm_stops_per_10k", "norm_walk_inv", "norm_ramp_cov",
               "norm_terrain_inv"]
    labels = ["Stop\nDensity", "Route\nDensity",
              "Per-Capita\nAccess", "Walking\nProximity",
              "Ramp\nCoverage", "Terrain\nSuitability"]

    # Compute cluster means
    cluster_means = {}
    for label in CLUSTER_COLOURS:
        mask = df["cluster_label"] == label
        if mask.any():
            cluster_means[label] = df.loc[mask, metrics].mean().values

    # Radar chart setup
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for label, values in cluster_means.items():
        vals = values.tolist() + [values[0]]   # close
        ax.plot(angles, vals, "o-", linewidth=2, markersize=6,
                color=CLUSTER_COLOURS[label], label=label)
        ax.fill(angles, vals, alpha=0.15, color=CLUSTER_COLOURS[label])

    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("Cluster Metric Profiles", fontsize=15,
                 fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

    path = os.path.join(OUTPUT_DIR, "cluster_profiles_radar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Radar chart saved: {path}")


# ===================================================================
# 4.  Summary report
# ===================================================================

def create_summary_report(df: pd.DataFrame) -> None:
    """
    Save a clean summary CSV and print key findings to the console.
    """
    print("  Creating summary report …")

    # Keep only the most important columns
    report_cols = [
        "rank", "district", "cluster_label", "accessibility_score",
        "total_stops", "kmb_stops", "citybus_stops", "nlb_stops",
        "population", "area_km2",
        "stops_per_km2", "routes_per_km2", "stops_per_10k",
        "avg_walking_dist_m", "pct_within_400m",
        "ramp_coverage_pct", "terrain_ruggedness", "operator_diversity",
    ]
    # Only include columns that actually exist
    report_cols = [c for c in report_cols if c in df.columns]
    report = df[report_cols].sort_values("rank")

    path = os.path.join(OUTPUT_DIR, "summary_report.csv")
    report.to_csv(path, index=False)
    print(f"  ✓ Report saved: {path}")

    # Print key findings
    print("\n" + "=" * 60)


# ===================================================================
# 5.  3D Visualisations
# ===================================================================

def create_3d_district_visual(df: pd.DataFrame) -> None:
    """
    Create an interactive 3D district-level scatter plot.

    Axes are selected to show service intensity (x/y) and overall
    accessibility (z), with marker size reflecting population.
    """
    print("  Creating 3D district visualisation …")

    needed = {"district", "stops_per_km2", "routes_per_km2", "accessibility_score", "cluster_label"}
    if not needed.issubset(df.columns):
        print("  ! Skipping 3D district chart: required columns missing")
        return

    plot_df = df.copy()
    if "population" not in plot_df.columns:
        plot_df["population"] = 1

    fig = px.scatter_3d(
        plot_df,
        x="stops_per_km2",
        y="routes_per_km2",
        z="accessibility_score",
        color="cluster_label",
        color_discrete_map=CLUSTER_COLOURS,
        size="population",
        size_max=38,
        hover_name="district",
        hover_data={
            "stops_per_10k": ":.2f" if "stops_per_10k" in plot_df.columns else False,
            "avg_walking_dist_m": ":.1f" if "avg_walking_dist_m" in plot_df.columns else False,
            "pct_within_400m": ":.1f" if "pct_within_400m" in plot_df.columns else False,
            "population": ":,.0f",
        },
        title="Hong Kong District Accessibility Landscape (3D)",
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="Stops per km^2",
            yaxis_title="Routes per km^2",
            zaxis_title="Accessibility Score",
        ),
        legend_title="Accessibility Cluster",
        margin=dict(l=0, r=0, b=0, t=48),
    )

    path = os.path.join(OUTPUT_DIR, "district_accessibility_3d.html")
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"  ✓ 3D district chart saved: {path}")


def create_3d_topography_visual(topography_path: str = TOPOGRAPHY_PATH) -> None:
    """
    Create an interactive 3D terrain scatter of sampled elevation points.

    This visual supports interpretation of how topography can amplify
    effective walking burden in steep areas.
    """
    print("  Creating 3D topography visualisation …")

    if not os.path.exists(topography_path):
        print("  ! Skipping 3D topography chart: topography file not found")
        return

    topo = pd.read_csv(topography_path)
    expected = {"lat", "lon", "elevation_m"}
    if topo.empty or not expected.issubset(topo.columns):
        print("  ! Skipping 3D topography chart: invalid topography schema")
        return

    topo = topo.dropna(subset=["lat", "lon", "elevation_m"])
    if topo.empty:
        print("  ! Skipping 3D topography chart: no valid elevation points")
        return

    sample_limit = 12000
    if len(topo) > sample_limit:
        topo = topo.sample(sample_limit, random_state=42)

    fig = px.scatter_3d(
        topo,
        x="lon",
        y="lat",
        z="elevation_m",
        color="elevation_m",
        color_continuous_scale="Viridis",
        opacity=0.75,
        title="Hong Kong Topography (3D Elevation Sample)",
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            zaxis_title="Elevation (m)",
        ),
        margin=dict(l=0, r=0, b=0, t=48),
    )

    path = os.path.join(OUTPUT_DIR, "topography_3d.html")
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"  ✓ 3D topography chart saved: {path}")


def create_3d_micro_accessibility_visual(micro_path: str = MICRO_GRID_PATH) -> None:
    """
    Create an interactive 3D micro-grid view where z is effective walk
    distance, highlighting pockets of local accessibility burden.
    """
    print("  Creating 3D micro-grid accessibility visualisation …")

    if not os.path.exists(micro_path):
        print("  ! Skipping 3D micro-grid chart: micro grid file not found")
        return

    micro = pd.read_csv(micro_path)
    expected = {"district", "lon", "lat", "effective_walk_dist_m", "cell_population"}
    if micro.empty or not expected.issubset(micro.columns):
        print("  ! Skipping 3D micro-grid chart: invalid micro-grid schema")
        return

    micro = micro.dropna(subset=["lon", "lat", "effective_walk_dist_m", "cell_population"])
    if micro.empty:
        print("  ! Skipping 3D micro-grid chart: no valid micro-grid records")
        return

    sample_limit = 18000
    if len(micro) > sample_limit:
        # Weighted sample keeps high-population cells more likely.
        p = micro["cell_population"].values + 1e-9
        p = p / p.sum()
        micro = micro.sample(sample_limit, random_state=42, weights=p)

    fig = px.scatter_3d(
        micro,
        x="lon",
        y="lat",
        z="effective_walk_dist_m",
        color="district",
        size="cell_population",
        size_max=12,
        opacity=0.62,
        title="Micro-Grid Effective Walking Burden (3D)",
        hover_data={
            "nearest_stop_dist_m": ":.1f" if "nearest_stop_dist_m" in micro.columns else False,
            "terrain_penalty": ":.3f" if "terrain_penalty" in micro.columns else False,
            "nearest_ramp_dist_m": ":.1f" if "nearest_ramp_dist_m" in micro.columns else False,
            "cell_population": ":.2f",
        },
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            zaxis_title="Effective Walk Distance (m)",
        ),
        margin=dict(l=0, r=0, b=0, t=48),
    )

    path = os.path.join(OUTPUT_DIR, "micro_accessibility_3d.html")
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"  ✓ 3D micro-grid chart saved: {path}")
    print("KEY FINDINGS")
    print("=" * 60)

    # Read Gini
    gini_path = os.path.join(OUTPUT_DIR, "gini_coefficient.txt")
    if os.path.exists(gini_path):
        with open(gini_path) as f:
            print(f"\n  {f.read().strip()}")

    # Best and worst districts
    best = df.loc[df["accessibility_score"].idxmax()]
    worst = df.loc[df["accessibility_score"].idxmin()]
    print(f"\n  Most accessible district:   {best['district']} "
          f"(score: {best['accessibility_score']:.3f})")
    print(f"  Least accessible district:  {worst['district']} "
          f"(score: {worst['accessibility_score']:.3f})")

    ratio = best["accessibility_score"] / (worst["accessibility_score"] + 1e-9)
    print(f"  Inequality ratio:           {ratio:.1f}×")

    # Coverage stats
    if "pct_within_400m" in df.columns:
        avg_400 = df["pct_within_400m"].mean()
        print(f"\n  Average % within 400 m of a stop: {avg_400:.1f}%")

    # Cluster counts
    print("\n  Cluster distribution:")
    for label in CLUSTER_COLOURS:
        count = (df["cluster_label"] == label).sum()
        districts = df.loc[df["cluster_label"] == label,
                           "district"].tolist()
        print(f"    {label}: {count} districts — "
              f"{', '.join(districts)}")

    print("\n" + "=" * 60)


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    print("=" * 60)
    print("05_visualise_results  ·  Generating maps and charts")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load clustered data
    data_path = os.path.join(OUTPUT_DIR, "clustered_districts.csv")
    df = pd.read_csv(data_path)
    print(f"\nLoaded {len(df)} districts with cluster labels\n")

    districts_path = os.path.join(RAW_DIR, "district_boundaries.json")

    create_choropleth(df, districts_path)
    create_bar_chart(df)
    create_radar_chart(df)
    create_3d_district_visual(df)
    create_3d_topography_visual()
    create_3d_micro_accessibility_visual()
    create_summary_report(df)

    print(f"\n{'=' * 60}")
    print("All visualisations generated.  Open accessibility_map.html")
    print("in a browser to explore the interactive map.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

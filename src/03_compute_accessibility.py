"""
03_compute_accessibility.py — Calculate transport accessibility scores
=====================================================================

This is where the real analysis happens.  We take the per-district
data from step 02 and compute several accessibility metrics, then
combine them into a single composite score.

Metrics computed
----------------
1.  stops_per_km²       → How densely served is the area?
2.  routes_per_km²      → How many different connections are available?
3.  stops_per_10k_pop   → Per-capita transport access (equity lens).
4.  avg_walking_dist_m  → Estimated average distance (metres) a
                          resident would need to walk to reach the
                          nearest bus or tram stop (grid-based).

We also compute:
-  A **composite accessibility score** (weighted average of the
   normalised metrics).
-  A **Gini coefficient** to quantify how unequally accessibility
   is distributed across the city.

The grid-based walking distance is calculated by overlaying a 200 m
grid across each district, finding the nearest transport stop for
every grid cell, and taking the mean.  This is more realistic than
just dividing area by stop count, because stop placement is not
uniform.

Output
------
    output/accessibility_scores.csv

Author:  CCAI-9012 Group E
Date:    March 2026
"""

import json
import os
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR    = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


# ===================================================================
# Distance utilities
# ===================================================================

def haversine_np(lat1: np.ndarray, lon1: np.ndarray,
                 lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """
    Vectorised Haversine formula — returns distance in metres.

    This is much faster than looping because NumPy computes the trig
    functions on entire arrays at once.

    Parameters
    ----------
    lat1, lon1 : arrays of point-set A (e.g. grid cells)
    lat2, lon2 : arrays of point-set B (e.g. stops)

    Returns
    -------
    distance in metres (same shape as inputs)
    """
    R = 6_371_000  # Earth's radius in metres

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + (
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    return R * 2 * np.arcsin(np.sqrt(a))


def compute_grid_walking_distance(district_gdf: gpd.GeoDataFrame,
                                   stops_gdf: gpd.GeoDataFrame,
                                   cell_size: float = 0.002
                                   ) -> pd.DataFrame:
    """
    For each district, lay a grid of ~200 m cells, then compute the
    average distance from each cell to the nearest transport stop.

    Parameters
    ----------
    district_gdf : GeoDataFrame with 'district' and 'geometry'
    stops_gdf    : GeoDataFrame with stop points
    cell_size    : grid spacing in degrees (~0.002° ≈ 200 m at HK's
                   latitude).

    Returns
    -------
    DataFrame with columns [district, avg_walking_dist_m, pct_within_400m]
    """
    results = []

    # Pre-extract stop coordinates as arrays for fast distance computation
    stop_lats = stops_gdf.geometry.y.values
    stop_lons = stops_gdf.geometry.x.values

    for _, row in district_gdf.iterrows():
        district_name = row["district"]
        polygon = row["geometry"]
        bounds  = polygon.bounds   # (minx, miny, maxx, maxy)

        # Build a grid of candidate points inside this district
        xs = np.arange(bounds[0], bounds[2], cell_size)
        ys = np.arange(bounds[1], bounds[3], cell_size)
        grid_x, grid_y = np.meshgrid(xs, ys)
        grid_x = grid_x.ravel()
        grid_y = grid_y.ravel()

        # Filter to points actually inside the district polygon
        # (using vectorised Shapely contains for efficiency)
        grid_points = gpd.GeoSeries(
            [Point(x, y) for x, y in zip(grid_x, grid_y)],
            crs="EPSG:4326"
        )
        inside_mask = grid_points.within(polygon)
        grid_x = grid_x[inside_mask.values]
        grid_y = grid_y[inside_mask.values]

        if len(grid_x) == 0:
            # Very small district — use centroid
            centroid = polygon.centroid
            grid_x = np.array([centroid.x])
            grid_y = np.array([centroid.y])

        # For each grid cell, find the distance to the nearest stop.
        # We compute all pairwise distances (grid cells × stops) and
        # take the column-wise minimum.  This sounds expensive but
        # NumPy broadcasting makes it surprisingly fast for our data
        # sizes (~thousands of cells × ~4,000 stops).
        min_dists = np.full(len(grid_x), np.inf)

        # Process stops in chunks to manage memory on smaller machines
        chunk_size = 500
        for i in range(0, len(stop_lats), chunk_size):
            s_lat = stop_lats[i:i + chunk_size]
            s_lon = stop_lons[i:i + chunk_size]

            # Broadcasting: grid_y[:, None] vs s_lat[None, :]
            dists = haversine_np(
                grid_y[:, None], grid_x[:, None],
                s_lat[None, :], s_lon[None, :]
            )
            chunk_min = dists.min(axis=1)
            min_dists = np.minimum(min_dists, chunk_min)

        avg_dist = float(np.mean(min_dists))
        pct_400  = float(np.mean(min_dists <= 400) * 100)

        results.append({
            "district":           district_name,
            "avg_walking_dist_m": round(avg_dist, 1),
            "pct_within_400m":    round(pct_400, 1),
        })

        print(f"    {district_name:25s}  "
              f"avg walk = {avg_dist:6.0f} m | "
              f"{pct_400:5.1f}% within 400 m")

    return pd.DataFrame(results)


# ===================================================================
# Normalisation and scoring
# ===================================================================

def min_max_normalise(series: pd.Series) -> pd.Series:
    """
    Scale a series to [0, 1] using min–max normalisation.

    We add a tiny epsilon to avoid division by zero in the edge case
    where all districts have the same value.
    """
    lo, hi = series.min(), series.max()
    return (series - lo) / (hi - lo + 1e-9)


def gini_coefficient(values: np.ndarray) -> float:
    """
    Compute the Gini coefficient of an array of values.

    The Gini coefficient measures inequality on a scale of 0 (perfect
    equality — every district identical) to 1 (maximum inequality —
    one district has everything, the rest have nothing).

    This is the same formula economists use for income inequality,
    but we apply it to transport accessibility.
    """
    values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return float(
        (2 * np.sum(index * values) - (n + 1) * np.sum(values))
        / (n * np.sum(values) + 1e-9)
    )


def compute_composite_score(df: pd.DataFrame,
                            weights: dict | None = None) -> pd.Series:
    """
    Combine normalised metrics into a single accessibility score.

    Default weights (configurable for sensitivity analysis):
        stop density   : 0.30
        route density  : 0.25
        per-capita     : 0.25
        walking distance : 0.20  (inverted — lower distance → higher score)

    We chose these weights after reviewing urban-planning literature.
    Stop density matters most because it directly determines how far
    people need to walk; route density captures connectivity; per-capita
    adjusts for fairness; walking distance gives the spatial reality.
    """
    if weights is None:
        weights = {
            "norm_stops_per_km2":   0.30,
            "norm_routes_per_km2":  0.25,
            "norm_stops_per_10k":   0.25,
            "norm_walk_inv":        0.20,
        }

    score = sum(df[col] * w for col, w in weights.items())
    return score


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    print("=" * 60)
    print("03_compute_accessibility  ·  Scoring districts")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load the processed data from step 02
    # ------------------------------------------------------------------
    data_path = os.path.join(OUTPUT_DIR, "district_transport_data.csv")
    df = pd.read_csv(data_path)
    print(f"\nLoaded {len(df)} districts from {data_path}")

    # ------------------------------------------------------------------
    # Basic density metrics
    # ------------------------------------------------------------------
    print("\n[1/3]  Computing density metrics …")

    df["stops_per_km2"]   = df["total_stops"]  / df["area_km2"]
    df["routes_per_km2"]  = df["total_routes"] / df["area_km2"]
    df["stops_per_10k"]   = df["total_stops"]  / df["population"] * 10_000

    # ------------------------------------------------------------------
    # Grid-based walking distance
    # ------------------------------------------------------------------
    print("\n[2/3]  Computing grid-based walking distances …")
    print("       (this takes a minute — overlaying a 200 m grid on each district)\n")

    # We need the stop locations and district polygons again
    districts = gpd.read_file(
        os.path.join(RAW_DIR, "district_boundaries.json"))

    # Find the district name column (same logic as 02_process_data)
    name_col = None
    for candidate in ["ENAME", "NAME_EN", "name_en", "DCNAME_EN",
                      "ename", "Name", "name", "DISTRICT"]:
        if candidate in districts.columns:
            name_col = candidate
            break
    if name_col:
        districts = districts.rename(columns={name_col: "district"})

    # Load bus stops
    with open(os.path.join(RAW_DIR, "kmb_bus_stops.json"), "r",
              encoding="utf-8") as f:
        bus_raw = json.load(f)
    bus_records = bus_raw.get("data", [])
    bus_lats = [float(r["lat"]) for r in bus_records
                if r.get("lat") and r.get("long")]
    bus_lons = [float(r["long"]) for r in bus_records
                if r.get("lat") and r.get("long")]

    # Load tram stops
    tram_path = os.path.join(RAW_DIR, "tram_stops.csv")
    tram_df = pd.read_csv(tram_path, encoding="utf-8")
    tram_df.columns = [c.strip().lower().replace(" ", "_") for c in tram_df.columns]
    lat_col  = next((c for c in tram_df.columns if "lat" in c), None)
    long_col = next((c for c in tram_df.columns if "lon" in c or "lng" in c), None)
    tram_lats, tram_lons = [], []
    if lat_col and long_col:
        tram_lats = tram_df[lat_col].dropna().astype(float).tolist()
        tram_lons = tram_df[long_col].dropna().astype(float).tolist()

    all_lats = bus_lats + tram_lats
    all_lons = bus_lons + tram_lons
    stop_points = gpd.GeoDataFrame(
        geometry=[Point(lon, lat) for lat, lon in zip(all_lats, all_lons)],
        crs="EPSG:4326"
    )

    walk_df = compute_grid_walking_distance(districts, stop_points)

    # Merge walking distances back
    # Fuzzy merge by normalised district name (same approach as step 02)
    def norm(s):
        return s.strip().lower().replace("&", "and").replace("  ", " ")

    df["_key"]      = df["district"].apply(norm)
    walk_df["_key"] = walk_df["district"].apply(norm)
    df = df.merge(walk_df[["_key", "avg_walking_dist_m", "pct_within_400m"]],
                  on="_key", how="left")
    df = df.drop(columns=["_key"])

    # ------------------------------------------------------------------
    # Normalise and combine into composite score
    # ------------------------------------------------------------------
    print("\n[3/3]  Computing composite accessibility score …\n")

    df["norm_stops_per_km2"]  = min_max_normalise(df["stops_per_km2"])
    df["norm_routes_per_km2"] = min_max_normalise(df["routes_per_km2"])
    df["norm_stops_per_10k"]  = min_max_normalise(df["stops_per_10k"])
    # Walking distance is inverted: shorter walk → higher score
    df["norm_walk_inv"] = min_max_normalise(
        df["avg_walking_dist_m"].max() - df["avg_walking_dist_m"]
    )

    df["accessibility_score"] = compute_composite_score(df)

    # Rank districts
    df = df.sort_values("accessibility_score", ascending=False)
    df["rank"] = range(1, len(df) + 1)

    # ------------------------------------------------------------------
    # Gini coefficient — a single number summarising inequality
    # ------------------------------------------------------------------
    gini = gini_coefficient(df["accessibility_score"].values)
    print(f"  Accessibility Gini coefficient: {gini:.4f}")
    print(f"  (0 = perfect equality, 1 = total inequality)\n")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path = os.path.join(OUTPUT_DIR, "accessibility_scores.csv")
    df.to_csv(out_path, index=False)

    # Also save the Gini value as a separate tiny file for easy reference
    gini_path = os.path.join(OUTPUT_DIR, "gini_coefficient.txt")
    with open(gini_path, "w") as f:
        f.write(f"Accessibility Gini Coefficient: {gini:.4f}\n")
        f.write(f"Computed over {len(df)} districts\n")

    print(f"{'=' * 60}")
    print(f"Scores saved to: {os.path.abspath(out_path)}")
    print(f"{'=' * 60}")

    # Print a quick summary table
    print(f"\n{'District':<25s} {'Score':>8s} {'Walk (m)':>10s} "
          f"{'<400m %':>8s} {'Rank':>5s}")
    print("-" * 60)
    for _, row in df.iterrows():
        print(f"{row['district']:<25s} "
              f"{row['accessibility_score']:>8.3f} "
              f"{row['avg_walking_dist_m']:>10.0f} "
              f"{row['pct_within_400m']:>8.1f} "
              f"{row['rank']:>5d}")


if __name__ == "__main__":
    main()

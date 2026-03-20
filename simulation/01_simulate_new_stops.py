"""
01_simulate_new_stops.py — Simulate candidate new bus-stop locations
===================================================================

This module proposes candidate bus-stop locations from micro-grid
accessibility outputs and classifies them into priority tiers.

Inputs
------
1. output/micro_accessibility_grid.csv
2. data/raw/district_boundaries.json
3. Existing stop datasets in data/raw/

Outputs
-------
1. simulation/output/candidate_new_bus_stops.csv
2. simulation/output/candidate_priority_summary.csv
3. simulation/output/new_stop_candidates_map.html

Author:  CCAI-9012 Group E
Date:    March 2026
"""

import json
import os

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
SIM_OUTPUT_DIR = os.path.join(BASE_DIR, "simulation", "output")


def load_existing_stops() -> pd.DataFrame:
    """Load and combine operator stop tables for map context and spacing checks."""
    with open(os.path.join(RAW_DIR, "kmb_bus_stops.json"), "r", encoding="utf-8") as f:
        kmb = json.load(f)
    with open(os.path.join(RAW_DIR, "citybus_stops.json"), "r", encoding="utf-8") as f:
        city = json.load(f)
    with open(os.path.join(RAW_DIR, "nlb_stops.json"), "r", encoding="utf-8") as f:
        nlb = json.load(f)

    kmb_df = pd.DataFrame(kmb.get("data", []))
    if not kmb_df.empty:
        kmb_df = kmb_df.rename(columns={"stop": "stop_id", "lat": "lat", "long": "lon"})
        kmb_df["operator"] = "KMB"
        kmb_df = kmb_df[["stop_id", "lat", "lon", "operator"]]

    city_df = pd.DataFrame(city.get("data", []))
    if not city_df.empty:
        city_df = city_df.rename(columns={"stop": "stop_id", "lat": "lat", "long": "lon"})
        city_df["operator"] = "Citybus"
        city_df = city_df[["stop_id", "lat", "lon", "operator"]]

    nlb_df = pd.DataFrame(nlb.get("data", []))
    if not nlb_df.empty:
        nlb_df = nlb_df.rename(columns={"stopId": "stop_id", "latitude": "lat", "longitude": "lon"})
        nlb_df["operator"] = "NLB"
        nlb_df = nlb_df[["stop_id", "lat", "lon", "operator"]]

    all_stops = pd.concat([kmb_df, city_df, nlb_df], ignore_index=True)
    all_stops["lat"] = pd.to_numeric(all_stops["lat"], errors="coerce")
    all_stops["lon"] = pd.to_numeric(all_stops["lon"], errors="coerce")
    all_stops = all_stops.dropna(subset=["lat", "lon"])
    return all_stops.drop_duplicates(subset=["operator", "stop_id", "lat", "lon"])


def haversine_np(lat1, lon1, lat2, lon2):
    """Vectorised Haversine distance in metres."""
    R = 6_371_000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def greedy_spacing_filter(candidates: pd.DataFrame, min_spacing_m: float = 350.0) -> pd.DataFrame:
    """Select high-score candidates while enforcing minimum spacing between selected points."""
    selected = []

    for _, row in candidates.sort_values("priority_score", ascending=False).iterrows():
        if not selected:
            selected.append(row)
            continue

        sel_lat = np.array([r["lat"] for r in selected])
        sel_lon = np.array([r["lon"] for r in selected])
        d = haversine_np(np.array([row["lat"]]), np.array([row["lon"]]), sel_lat, sel_lon)
        if float(d.min()) >= min_spacing_m:
            selected.append(row)

    if not selected:
        return candidates.head(0).copy()

    return pd.DataFrame(selected)


def assign_priority(score_series: pd.Series) -> pd.Series:
    """Map continuous scores to high / medium / low priority tiers."""
    q_high = score_series.quantile(0.67)
    q_med = score_series.quantile(0.33)

    def _label(x):
        if x >= q_high:
            return "High Priority"
        if x >= q_med:
            return "Medium Priority"
        return "Low Priority"

    return score_series.apply(_label)


def create_candidate_map(candidates: pd.DataFrame, existing_stops: pd.DataFrame) -> folium.Map:
    """Create interactive map showing existing stops and candidate tiers."""
    m = folium.Map(location=[22.35, 114.15], zoom_start=11, tiles="CartoDB positron")

    # Existing stops are shown as subtle blue dots to avoid visual clutter.
    for _, s in existing_stops.iterrows():
        folium.CircleMarker(
            location=[s["lat"], s["lon"]],
            radius=1,
            color="#60a5fa",
            fill=True,
            fill_opacity=0.35,
            weight=0,
        ).add_to(m)

    colors = {
        "High Priority": "#dc2626",
        "Medium Priority": "#f59e0b",
        "Low Priority": "#22c55e",
    }

    for _, c in candidates.iterrows():
        folium.CircleMarker(
            location=[c["lat"], c["lon"]],
            radius=5,
            color=colors.get(c["priority_level"], "#6b7280"),
            fill=True,
            fill_opacity=0.9,
            popup=(
                f"District: {c['district']}<br>"
                f"Priority: {c['priority_level']}<br>"
                f"Score: {c['priority_score']:.2f}<br>"
                f"Eff. Walk Dist: {c['effective_walk_dist_m']:.0f} m<br>"
                f"Cell Population: {c['cell_population']:.1f}"
            ),
        ).add_to(m)

    return m


def main() -> None:
    print("=" * 60)
    print("Simulation  ·  Candidate new bus-stop location modelling")
    print("=" * 60)

    os.makedirs(SIM_OUTPUT_DIR, exist_ok=True)

    micro_path = os.path.join(OUTPUT_DIR, "micro_accessibility_grid.csv")
    if not os.path.exists(micro_path):
        raise FileNotFoundError(
            "Missing micro_accessibility_grid.csv. Run src/03_compute_accessibility.py first."
        )

    micro = pd.read_csv(micro_path)
    existing = load_existing_stops()

    # Candidate scoring emphasises underserved cells with higher population burden.
    # The 350m threshold reflects practical walking-distance expectations.
    burden = np.maximum(micro["effective_walk_dist_m"] - 350.0, 0.0)
    micro["priority_score"] = burden * np.sqrt(np.maximum(micro["cell_population"], 0.0))

    # Keep only underserved cells, then take strongest cells before spacing filter.
    base_candidates = micro[micro["priority_score"] > 0].copy()
    base_candidates = base_candidates.sort_values("priority_score", ascending=False).head(2000)

    spaced_candidates = greedy_spacing_filter(base_candidates, min_spacing_m=350.0)
    spaced_candidates = spaced_candidates.head(250).copy()

    if spaced_candidates.empty:
        print("No candidate points were generated. Check upstream accessibility inputs.")
        return

    spaced_candidates["priority_level"] = assign_priority(spaced_candidates["priority_score"])
    spaced_candidates["candidate_id"] = [f"CAND_{i:04d}" for i in range(1, len(spaced_candidates) + 1)]

    # Order columns for clean output consumption.
    ordered_cols = [
        "candidate_id",
        "district",
        "lat",
        "lon",
        "priority_level",
        "priority_score",
        "effective_walk_dist_m",
        "nearest_stop_dist_m",
        "cell_population",
        "terrain_penalty",
        "nearest_ramp_dist_m",
    ]
    candidates_out = spaced_candidates[ordered_cols].sort_values(
        ["priority_level", "priority_score"],
        ascending=[True, False],
    )

    summary = (
        candidates_out.groupby(["district", "priority_level"])
        .size()
        .reset_index(name="candidate_count")
        .sort_values(["priority_level", "candidate_count"], ascending=[True, False])
    )

    candidates_csv = os.path.join(SIM_OUTPUT_DIR, "candidate_new_bus_stops.csv")
    summary_csv = os.path.join(SIM_OUTPUT_DIR, "candidate_priority_summary.csv")
    map_html = os.path.join(SIM_OUTPUT_DIR, "new_stop_candidates_map.html")

    candidates_out.to_csv(candidates_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    m = create_candidate_map(candidates_out, existing)
    m.save(map_html)

    print(f"Candidate locations saved: {candidates_csv}")
    print(f"Priority summary saved:    {summary_csv}")
    print(f"Interactive map saved:     {map_html}")
    print("=" * 60)


if __name__ == "__main__":
    main()

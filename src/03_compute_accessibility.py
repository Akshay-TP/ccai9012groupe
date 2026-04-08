"""
03_compute_accessibility.py — Calculate transport accessibility scores
=====================================================================

This script computes district accessibility using road-network shortest
paths instead of straight-line (Haversine) distance.

Core modelling updates
----------------------
1. Grid-cell to stop distance is measured along the road graph only.
2. Ramp proximity is measured along the same road graph.
3. Terrain and ramp factors still adjust effective walking burden.
4. Remote-area handling is configurable:
   - exclude_remote (recommended)
   - distance_cap

Outputs
-------
1. output/accessibility_scores.csv
2. output/gini_coefficient.txt
3. output/micro_accessibility_grid.csv

Author:  CCAI-9012 Group E
Date:    April 2026
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from road_network import (
    build_node_balltree,
    graph_node_arrays,
    load_road_graph,
    lookup_distances,
    multi_source_shortest_path_lengths,
    snap_points_to_nodes,
    to_simple_undirected,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure Unicode status logs render on Windows terminals.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


def haversine_np(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorised Haversine formula that returns distance in metres."""
    R = 6_371_000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + (np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))


def min_max_normalise(series: pd.Series) -> pd.Series:
    """Scale a series to [0, 1] with numerical safety."""
    lo, hi = series.min(), series.max()
    return (series - lo) / (hi - lo + 1e-9)


def gini_coefficient(values: np.ndarray) -> float:
    """Compute the Gini coefficient of an array of positive values."""
    values = np.sort(values)
    n = len(values)
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * values) - (n + 1) * np.sum(values)) / (n * np.sum(values) + 1e-9))


def load_districts() -> gpd.GeoDataFrame:
    """Load district polygons and normalise district name column."""
    gdf = gpd.read_file(os.path.join(RAW_DIR, "district_boundaries.json"))
    name_col = None
    for candidate in [
        "ENAME",
        "NAME_EN",
        "name_en",
        "DCNAME_EN",
        "ename",
        "Name",
        "name",
        "DISTRICT",
        "District",
    ]:
        if candidate in gdf.columns:
            name_col = candidate
            break

    if name_col is None:
        raise KeyError(f"No district name column found in boundary file: {gdf.columns.tolist()}")

    return gdf.rename(columns={name_col: "district"})[["district", "geometry"]]


def load_all_bus_stops() -> gpd.GeoDataFrame:
    """Load KMB, Citybus, and NLB stops and return one GeoDataFrame."""
    with open(os.path.join(RAW_DIR, "kmb_bus_stops.json"), "r", encoding="utf-8") as f:
        kmb = json.load(f)
    with open(os.path.join(RAW_DIR, "citybus_stops.json"), "r", encoding="utf-8") as f:
        city = json.load(f)
    with open(os.path.join(RAW_DIR, "nlb_stops.json"), "r", encoding="utf-8") as f:
        nlb = json.load(f)

    kmb_df = pd.DataFrame(kmb.get("data", []))
    if not kmb_df.empty:
        kmb_df = kmb_df.rename(columns={"stop": "stop_id", "lat": "lat", "long": "long"})
        kmb_df["operator"] = "KMB"
        kmb_df = kmb_df[["stop_id", "lat", "long", "operator"]]

    city_df = pd.DataFrame(city.get("data", []))
    if not city_df.empty:
        city_df = city_df.rename(columns={"stop": "stop_id", "lat": "lat", "long": "long"})
        city_df["operator"] = "Citybus"
        city_df = city_df[["stop_id", "lat", "long", "operator"]]

    nlb_df = pd.DataFrame(nlb.get("data", []))
    if not nlb_df.empty:
        nlb_df = nlb_df.rename(columns={"stopId": "stop_id", "latitude": "lat", "longitude": "long"})
        nlb_df["operator"] = "NLB"
        nlb_df = nlb_df[["stop_id", "lat", "long", "operator"]]

    parts = [df for df in [kmb_df, city_df, nlb_df] if not df.empty]
    all_stops = pd.concat(parts, ignore_index=True)
    all_stops["lat"] = pd.to_numeric(all_stops["lat"], errors="coerce")
    all_stops["long"] = pd.to_numeric(all_stops["long"], errors="coerce")
    all_stops = all_stops.dropna(subset=["lat", "long"])
    all_stops = all_stops.drop_duplicates(subset=["operator", "stop_id", "lat", "long"])

    return gpd.GeoDataFrame(
        all_stops,
        geometry=[Point(lon, lat) for lat, lon in zip(all_stops["lat"], all_stops["long"])],
        crs="EPSG:4326",
    )


def load_ramps() -> gpd.GeoDataFrame:
    """Load ramp proxy points if available."""
    path = os.path.join(RAW_DIR, "hk_ramps_points.csv")
    if not os.path.exists(path):
        return gpd.GeoDataFrame(columns=["lat", "lon", "geometry"], geometry="geometry", crs="EPSG:4326")

    df = pd.read_csv(path)
    if df.empty:
        return gpd.GeoDataFrame(columns=["lat", "lon", "geometry"], geometry="geometry", crs="EPSG:4326")

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    return gpd.GeoDataFrame(
        df,
        geometry=[Point(lon, lat) for lat, lon in zip(df["lat"], df["lon"])],
        crs="EPSG:4326",
    )


def load_topography_points() -> pd.DataFrame:
    """Load sampled topography points if available."""
    path = os.path.join(RAW_DIR, "hk_topography_points.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["lat", "lon", "elevation_m"])

    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["lat", "lon", "elevation_m"])

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["elevation_m"] = pd.to_numeric(df["elevation_m"], errors="coerce")
    return df.dropna(subset=["lat", "lon", "elevation_m"])


def estimate_terrain_penalty(
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    topo_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate terrain ruggedness and multiplicative penalty at each grid point.
    """
    if topo_df.empty:
        n = len(grid_lat)
        return np.zeros(n), np.ones(n)

    t_lat = topo_df["lat"].values
    t_lon = topo_df["lon"].values
    t_ele = topo_df["elevation_m"].values

    ruggedness = np.zeros(len(grid_lat))

    chunk = 500
    for i in range(0, len(grid_lat), chunk):
        glat = grid_lat[i:i + chunk]
        glon = grid_lon[i:i + chunk]
        d = haversine_np(glat[:, None], glon[:, None], t_lat[None, :], t_lon[None, :])

        k = min(6, d.shape[1])
        idx = np.argpartition(d, k - 1, axis=1)[:, :k]
        local_ele = t_ele[idx]
        ruggedness[i:i + chunk] = np.std(local_ele, axis=1)

    terrain_penalty = 1.0 + (ruggedness / 50.0)
    terrain_penalty = np.clip(terrain_penalty, 1.0, 2.5)
    return ruggedness, terrain_penalty


def normalise_key(s: str) -> str:
    """Normalise district names for safer joins."""
    return s.strip().lower().replace("&", "and").replace("  ", " ")


def apply_remote_policy(
    road_dist_m: np.ndarray,
    policy: str,
    distance_cap_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply remote-area handling policy to road distances.

    Returns:
    1) adjusted distance array
    2) remote mask before policy (unreachable or beyond cap)
    """
    remote_mask = (~np.isfinite(road_dist_m)) | (road_dist_m > distance_cap_m)

    if policy == "distance_cap":
        capped = np.where(np.isfinite(road_dist_m), road_dist_m, distance_cap_m)
        capped = np.minimum(capped, distance_cap_m)
        return capped, remote_mask

    if policy == "exclude_remote":
        return road_dist_m, remote_mask

    raise ValueError(f"Unknown remote policy: {policy}")


def compute_microgrid_metrics_road(
    district_gdf: gpd.GeoDataFrame,
    stops_gdf: gpd.GeoDataFrame,
    ramps_gdf: gpd.GeoDataFrame,
    topo_df: pd.DataFrame,
    district_population: dict,
    raw_dir: str,
    cell_size: float,
    remote_policy: str,
    distance_cap_m: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute district metrics from fine grid cells using road-network distance.

    Returns:
    1. district-level summary metrics
    2. micro-grid cell table for simulation and diagnostics
    """
    print("  ↳ Loading and indexing road graph …")
    graph_multi = load_road_graph(raw_dir)
    graph = to_simple_undirected(graph_multi)

    node_ids, node_lats, node_lons = graph_node_arrays(graph)
    balltree = build_node_balltree(node_lats, node_lons)

    print("  ↳ Snapping existing stops to road nodes …")
    stop_lats = stops_gdf.geometry.y.values
    stop_lons = stops_gdf.geometry.x.values
    stop_nodes, _ = snap_points_to_nodes(stop_lats, stop_lons, node_ids, balltree)
    stop_dist_map = multi_source_shortest_path_lengths(graph, np.unique(stop_nodes))

    if not ramps_gdf.empty:
        print("  ↳ Snapping ramp proxies to road nodes …")
        ramp_lats = ramps_gdf.geometry.y.values
        ramp_lons = ramps_gdf.geometry.x.values
        ramp_nodes, _ = snap_points_to_nodes(ramp_lats, ramp_lons, node_ids, balltree)
        ramp_dist_map = multi_source_shortest_path_lengths(graph, np.unique(ramp_nodes))
    else:
        ramp_dist_map = {}

    district_rows = []
    micro_rows = []

    for _, row in district_gdf.iterrows():
        district = row["district"]
        poly = row["geometry"]
        minx, miny, maxx, maxy = poly.bounds

        xs = np.arange(minx, maxx, cell_size)
        ys = np.arange(miny, maxy, cell_size)
        gx, gy = np.meshgrid(xs, ys)
        gx = gx.ravel()
        gy = gy.ravel()

        pts = gpd.GeoSeries([Point(x, y) for x, y in zip(gx, gy)], crs="EPSG:4326")
        mask = pts.within(poly)
        gx = gx[mask.values]
        gy = gy[mask.values]

        if len(gx) == 0:
            c = poly.centroid
            gx = np.array([c.x])
            gy = np.array([c.y])

        snap_nodes, connector_dist = snap_points_to_nodes(gy, gx, node_ids, balltree)
        road_stop_dist = lookup_distances(
            snapped_node_ids=snap_nodes,
            distance_map=stop_dist_map,
            connector_m=connector_dist,
            unreachable_value=np.inf,
        )

        road_ramp_dist = lookup_distances(
            snapped_node_ids=snap_nodes,
            distance_map=ramp_dist_map,
            connector_m=connector_dist,
            unreachable_value=np.inf,
        ) if ramp_dist_map else np.full(len(gx), np.inf)

        road_stop_dist, remote_mask = apply_remote_policy(
            road_dist_m=road_stop_dist,
            policy=remote_policy,
            distance_cap_m=distance_cap_m,
        )

        keep_mask = np.ones(len(gx), dtype=bool)
        if remote_policy == "exclude_remote":
            keep_mask = ~remote_mask
            if not keep_mask.any():
                # Keep one fallback point so the district remains represented.
                finite_idx = np.where(np.isfinite(road_stop_dist))[0]
                if len(finite_idx) > 0:
                    best_idx = finite_idx[np.argmin(road_stop_dist[finite_idx])]
                else:
                    best_idx = 0
                    road_stop_dist[best_idx] = distance_cap_m
                keep_mask = np.zeros(len(gx), dtype=bool)
                keep_mask[best_idx] = True

        gx_keep = gx[keep_mask]
        gy_keep = gy[keep_mask]
        stop_keep = road_stop_dist[keep_mask]
        ramp_keep = road_ramp_dist[keep_mask]
        remote_keep = remote_mask[keep_mask]

        ruggedness, terrain_penalty = estimate_terrain_penalty(gy_keep, gx_keep, topo_df)

        ramp_factor = np.where(ramp_keep <= 80.0, 0.88, 1.0)
        effective_dist = stop_keep * terrain_penalty * ramp_factor

        stop_adj = 1.0 / (1.0 + (stop_keep / 500.0))
        terrain_pref = 1.0 / terrain_penalty
        pop_weight_raw = 0.65 * stop_adj + 0.35 * terrain_pref
        pop_weight_raw = np.clip(pop_weight_raw, 1e-6, None)

        district_pop = float(district_population.get(normalise_key(district), 0.0))
        pop_share = pop_weight_raw / pop_weight_raw.sum()
        cell_pop = pop_share * district_pop

        weighted_avg_walk = float(np.sum(effective_dist * cell_pop) / (cell_pop.sum() + 1e-9))
        weighted_cov_400 = float(np.sum(cell_pop[effective_dist <= 400]) / (cell_pop.sum() + 1e-9) * 100)
        ramp_cov = float(np.sum(cell_pop[ramp_keep <= 80]) / (cell_pop.sum() + 1e-9) * 100)
        terrain_rug = float(np.average(ruggedness, weights=cell_pop + 1e-9))

        district_rows.append(
            {
                "district": district,
                "avg_walking_dist_m": round(weighted_avg_walk, 1),
                "pct_within_400m": round(weighted_cov_400, 1),
                "ramp_coverage_pct": round(ramp_cov, 1),
                "terrain_ruggedness": round(terrain_rug, 2),
                "micro_grid_cells": int(len(gx_keep)),
                "remote_cells_pct": round(float(remote_mask.mean() * 100.0), 2),
                "distance_mode": "road_network",
                "remote_policy": remote_policy,
            }
        )

        for x, y, d_road, d_eff, cp, tr, rd, is_remote in zip(
            gx_keep,
            gy_keep,
            stop_keep,
            effective_dist,
            cell_pop,
            terrain_penalty,
            ramp_keep,
            remote_keep,
        ):
            micro_rows.append(
                {
                    "district": district,
                    "lon": float(x),
                    "lat": float(y),
                    "nearest_stop_dist_m": float(d_road),
                    "road_nearest_stop_dist_m": float(d_road),
                    "effective_walk_dist_m": float(d_eff),
                    "cell_population": float(cp),
                    "terrain_penalty": float(tr),
                    "nearest_ramp_dist_m": float(rd),
                    "road_nearest_ramp_dist_m": float(rd),
                    "is_remote_cell": bool(is_remote),
                    "distance_mode": "road_network",
                    "remote_policy": remote_policy,
                }
            )

        print(
            f"    {district:25s} road avg walk={weighted_avg_walk:6.0f}m | "
            f"cov400={weighted_cov_400:5.1f}% | remote={remote_mask.mean() * 100:5.1f}%"
        )

    return pd.DataFrame(district_rows), pd.DataFrame(micro_rows)


def compute_composite_score(df: pd.DataFrame, weights: dict | None = None) -> pd.Series:
    """Combine normalised metrics into one accessibility score."""
    if weights is None:
        weights = {
            "norm_stops_per_km2": 0.20,
            "norm_routes_per_km2": 0.18,
            "norm_stops_per_10k": 0.17,
            "norm_walk_inv": 0.20,
            "norm_operator_div": 0.10,
            "norm_ramp_cov": 0.08,
            "norm_terrain_inv": 0.07,
        }
    return sum(df[c] * w for c, w in weights.items())


def main(remote_policy: str, distance_cap_m: float, cell_size: float) -> None:
    print("=" * 60)
    print("03_compute_accessibility  ·  Scoring districts (road-network)")
    print("=" * 60)

    data_path = os.path.join(OUTPUT_DIR, "district_transport_data.csv")
    df = pd.read_csv(data_path)
    print(f"\nLoaded {len(df)} districts from {data_path}")

    print("\n[1/4]  Computing baseline density metrics …")
    df["stops_per_km2"] = df["total_stops"] / df["area_km2"]
    df["routes_per_km2"] = df["total_routes"] / df["area_km2"]
    df["stops_per_10k"] = df["total_stops"] / df["population"] * 10_000

    print("\n[2/4]  Running road-network micro-grid accessibility model …")
    districts = load_districts()
    stops_gdf = load_all_bus_stops()
    ramps_gdf = load_ramps()
    topo_df = load_topography_points()

    pop_map = {
        normalise_key(str(d)): float(p)
        for d, p in zip(df["district"], df["population"])
    }
    district_grid, micro_grid = compute_microgrid_metrics_road(
        district_gdf=districts,
        stops_gdf=stops_gdf,
        ramps_gdf=ramps_gdf,
        topo_df=topo_df,
        district_population=pop_map,
        raw_dir=RAW_DIR,
        cell_size=cell_size,
        remote_policy=remote_policy,
        distance_cap_m=distance_cap_m,
    )

    df["_key"] = df["district"].apply(normalise_key)
    district_grid["_key"] = district_grid["district"].apply(normalise_key)
    df = df.merge(
        district_grid[
            [
                "_key",
                "avg_walking_dist_m",
                "pct_within_400m",
                "ramp_coverage_pct",
                "terrain_ruggedness",
                "micro_grid_cells",
                "remote_cells_pct",
                "distance_mode",
                "remote_policy",
            ]
        ],
        on="_key",
        how="left",
    ).drop(columns=["_key"])

    print("\n[3/4]  Normalising features and computing composite score …")
    df["norm_stops_per_km2"] = min_max_normalise(df["stops_per_km2"])
    df["norm_routes_per_km2"] = min_max_normalise(df["routes_per_km2"])
    df["norm_stops_per_10k"] = min_max_normalise(df["stops_per_10k"])
    df["norm_walk_inv"] = min_max_normalise(df["avg_walking_dist_m"].max() - df["avg_walking_dist_m"])
    df["norm_operator_div"] = min_max_normalise(df["operator_diversity"])
    df["norm_ramp_cov"] = min_max_normalise(df["ramp_coverage_pct"])
    df["norm_terrain_inv"] = min_max_normalise(df["terrain_ruggedness"].max() - df["terrain_ruggedness"])

    df["accessibility_score"] = compute_composite_score(df)
    df = df.sort_values("accessibility_score", ascending=False)
    df["rank"] = range(1, len(df) + 1)

    print("\n[4/4]  Saving accessibility outputs …")
    out_path = os.path.join(OUTPUT_DIR, "accessibility_scores.csv")
    micro_path = os.path.join(OUTPUT_DIR, "micro_accessibility_grid.csv")
    gini_path = os.path.join(OUTPUT_DIR, "gini_coefficient.txt")

    df.to_csv(out_path, index=False)
    micro_grid.to_csv(micro_path, index=False)

    gini = gini_coefficient(df["accessibility_score"].values)
    with open(gini_path, "w", encoding="utf-8") as f:
        f.write(f"Accessibility Gini Coefficient: {gini:.4f}\n")
        f.write(f"Computed over {len(df)} districts\n")
        f.write("Distance mode: road_network\n")
        f.write(f"Remote policy: {remote_policy}\n")
        f.write(f"Remote threshold/cap (m): {distance_cap_m:.0f}\n")

    print(f"  Accessibility Gini coefficient: {gini:.4f}")
    print("=" * 60)
    print(f"Scores saved to: {os.path.abspath(out_path)}")
    print(f"Micro-grid saved to: {os.path.abspath(micro_path)}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute accessibility using road-network shortest-path distances.",
    )
    parser.add_argument(
        "--remote-policy",
        choices=["exclude_remote", "distance_cap"],
        default="exclude_remote",
        help="How to handle unreachable/far-flung grid cells.",
    )
    parser.add_argument(
        "--distance-cap-m",
        type=float,
        default=2000.0,
        help="Remote threshold (exclude mode) or cap value (cap mode), in metres.",
    )
    parser.add_argument(
        "--grid-cell-size",
        type=float,
        default=0.001,
        help="Grid cell size in degrees (~0.001 is about 100 m).",
    )
    args = parser.parse_args()

    main(
        remote_policy=args.remote_policy,
        distance_cap_m=args.distance_cap_m,
        cell_size=args.grid_cell_size,
    )

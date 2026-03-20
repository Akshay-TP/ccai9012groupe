"""
02_process_data.py — Clean, merge, and spatially join the raw datasets
======================================================================

This script transforms raw API files into a district-level analytical
table for downstream accessibility scoring.

What it does
------------
1. Parse KMB, Citybus, and NLB stop datasets into one common schema.
2. Parse route-stop mappings and compute unique routes served per stop.
3. Spatially assign each stop to a district (point-in-polygon join).
4. Aggregate district-level transport metrics.
5. Merge with district population and area statistics.

Output
------
    output/district_transport_data.csv

Author:  CCAI-9012 Group E
Date:    March 2026
"""

import json
import os
import sys
import warnings

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


def load_json(filename: str) -> dict:
    """Read a JSON file from data/raw/."""
    path = os.path.join(RAW_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_kmb_stops(raw: dict) -> pd.DataFrame:
    """Parse KMB stops to standard columns."""
    df = pd.DataFrame(raw.get("data", []))
    if df.empty:
        return pd.DataFrame(columns=["stop_id", "stop_name", "lat", "long", "operator", "mode"])

    df = df.rename(columns={"stop": "stop_id", "name_en": "stop_name"})
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["long"] = pd.to_numeric(df["long"], errors="coerce")
    df = df.dropna(subset=["lat", "long"])
    df["operator"] = "KMB"
    df["mode"] = "bus"
    return df[["stop_id", "stop_name", "lat", "long", "operator", "mode"]]


def parse_citybus_stops(raw: dict) -> pd.DataFrame:
    """Parse Citybus stops to standard columns."""
    df = pd.DataFrame(raw.get("data", []))
    if df.empty:
        return pd.DataFrame(columns=["stop_id", "stop_name", "lat", "long", "operator", "mode"])

    df = df.rename(columns={"stop": "stop_id", "name_en": "stop_name"})
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["long"] = pd.to_numeric(df["long"], errors="coerce")
    df = df.dropna(subset=["lat", "long"])
    df["operator"] = "Citybus"
    df["mode"] = "bus"
    return df[["stop_id", "stop_name", "lat", "long", "operator", "mode"]]


def parse_nlb_stops(raw: dict) -> pd.DataFrame:
    """Parse New Lantao Bus stops to standard columns."""
    df = pd.DataFrame(raw.get("data", []))
    if df.empty:
        return pd.DataFrame(columns=["stop_id", "stop_name", "lat", "long", "operator", "mode"])

    df = df.rename(
        columns={
            "stopId": "stop_id",
            "stopName_e": "stop_name",
            "latitude": "lat",
            "longitude": "long",
        }
    )
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["long"] = pd.to_numeric(df["long"], errors="coerce")
    df = df.dropna(subset=["lat", "long"])
    df["operator"] = "NLB"
    df["mode"] = "bus"
    return df[["stop_id", "stop_name", "lat", "long", "operator", "mode"]]


def routes_per_stop_kmb(raw: dict) -> pd.DataFrame:
    """Compute KMB route count per stop."""
    df = pd.DataFrame(raw.get("data", []))
    if df.empty:
        return pd.DataFrame(columns=["operator", "stop_id", "routes_served"])

    # Route key includes direction/service type so service variants are counted.
    df["route_key"] = (
        df["route"].astype(str)
        + "|"
        + df.get("bound", "").astype(str)
        + "|"
        + df.get("service_type", "").astype(str)
    )
    out = (
        df.groupby("stop")["route_key"]
        .nunique()
        .reset_index()
        .rename(columns={"stop": "stop_id", "route_key": "routes_served"})
    )
    out["operator"] = "KMB"
    return out[["operator", "stop_id", "routes_served"]]


def routes_per_stop_citybus(raw: dict) -> pd.DataFrame:
    """Compute Citybus route count per stop."""
    df = pd.DataFrame(raw.get("data", []))
    if df.empty:
        return pd.DataFrame(columns=["operator", "stop_id", "routes_served"])

    df["route_key"] = (
        df.get("route", "").astype(str)
        + "|"
        + df.get("direction", "").astype(str)
        + "|"
        + df.get("service_type", "").astype(str)
    )
    out = (
        df.groupby("stop")["route_key"]
        .nunique()
        .reset_index()
        .rename(columns={"stop": "stop_id", "route_key": "routes_served"})
    )
    out["operator"] = "Citybus"
    return out[["operator", "stop_id", "routes_served"]]


def routes_per_stop_nlb(raw: dict) -> pd.DataFrame:
    """Compute NLB route count per stop."""
    df = pd.DataFrame(raw.get("data", []))
    if df.empty:
        return pd.DataFrame(columns=["operator", "stop_id", "routes_served"])

    df["route_key"] = df.get("routeNo", "").astype(str)
    out = (
        df.groupby("stopId")["route_key"]
        .nunique()
        .reset_index()
        .rename(columns={"stopId": "stop_id", "route_key": "routes_served"})
    )
    out["operator"] = "NLB"
    return out[["operator", "stop_id", "routes_served"]]


def load_districts(path: str) -> gpd.GeoDataFrame:
    """Load district polygons and standardise district name column."""
    gdf = gpd.read_file(path)
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
        print("Could not find district-name column in boundary GeoJSON.", file=sys.stderr)
        print(f"Columns found: {gdf.columns.tolist()}", file=sys.stderr)
        sys.exit(1)

    gdf = gdf.rename(columns={name_col: "district"})
    return gdf[["district", "geometry"]]


def load_population() -> pd.DataFrame:
    """Load curated population and area by district."""
    return pd.read_csv(os.path.join(DATA_DIR, "population_by_district.csv"))


def normalise_district_name(name: str) -> str:
    """Normalise district labels to reduce naming mismatch during merge."""
    return name.strip().lower().replace("&", "and").replace("  ", " ")


def fuzzy_merge_districts(stats: pd.DataFrame, pop: pd.DataFrame) -> pd.DataFrame:
    """Merge district-level stats with population using normalised join keys."""
    left = stats.copy()
    right = pop.copy()
    left["_key"] = left["district"].apply(normalise_district_name)
    right["_key"] = right["district"].apply(normalise_district_name)

    merged = left.merge(right, on="_key", how="outer", suffixes=("_geo", "_pop"))
    merged["district"] = merged["district_pop"].fillna(merged["district_geo"])
    merged = merged.drop(columns=["district_geo", "district_pop", "_key"], errors="ignore")
    return merged


def main() -> None:
    print("=" * 60)
    print("02_process_data  ·  Cleaning and merging datasets")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load and parse operator stop datasets
    # ------------------------------------------------------------------
    print("\n[1/6]  Parsing operator stop datasets …")
    kmb_stops = parse_kmb_stops(load_json("kmb_bus_stops.json"))
    city_stops = parse_citybus_stops(load_json("citybus_stops.json"))
    nlb_stops = parse_nlb_stops(load_json("nlb_stops.json"))

    all_stops = pd.concat([kmb_stops, city_stops, nlb_stops], ignore_index=True)
    all_stops["stop_id"] = all_stops["stop_id"].astype(str)
    all_stops = all_stops.drop_duplicates(subset=["operator", "stop_id", "lat", "long"])

    print(
        f"       → KMB={len(kmb_stops):,}, Citybus={len(city_stops):,}, NLB={len(nlb_stops):,}, "
        f"combined={len(all_stops):,}"
    )

    # ------------------------------------------------------------------
    # 2. Compute route counts per stop for each operator
    # ------------------------------------------------------------------
    print("\n[2/6]  Computing routes served per stop …")
    kmb_rs = routes_per_stop_kmb(load_json("kmb_route_stops.json"))
    city_rs = routes_per_stop_citybus(load_json("citybus_route_stops.json"))
    nlb_rs = routes_per_stop_nlb(load_json("nlb_route_stops.json"))

    all_rs = pd.concat([kmb_rs, city_rs, nlb_rs], ignore_index=True)
    all_rs["stop_id"] = all_rs["stop_id"].astype(str)

    all_stops = all_stops.merge(
        all_rs,
        on=["operator", "stop_id"],
        how="left",
    )
    all_stops["routes_served"] = all_stops["routes_served"].fillna(0).astype(int)
    print("       → Route counts attached")

    # ------------------------------------------------------------------
    # 3. Spatial join stops to district polygons
    # ------------------------------------------------------------------
    print("\n[3/6]  Spatial join — assigning stops to districts …")
    stops_gdf = gpd.GeoDataFrame(
        all_stops,
        geometry=[Point(xy) for xy in zip(all_stops["long"], all_stops["lat"])],
        crs="EPSG:4326",
    )
    districts = load_districts(os.path.join(RAW_DIR, "district_boundaries.json"))
    joined = gpd.sjoin(stops_gdf, districts, how="left", predicate="within")

    n_unassigned = int(joined["district"].isna().sum())
    if n_unassigned:
        print(f"       ⚠ {n_unassigned} stops are outside district polygons and will be ignored")
    joined = joined.dropna(subset=["district"])  # Keep only district-assigned stops.

    # ------------------------------------------------------------------
    # 4. Aggregate district-level metrics
    # ------------------------------------------------------------------
    print("\n[4/6]  Aggregating district-level metrics …")
    district_stats = (
        joined.groupby("district")
        .agg(
            total_stops=("stop_id", "count"),
            kmb_stops=("operator", lambda x: (x == "KMB").sum()),
            citybus_stops=("operator", lambda x: (x == "Citybus").sum()),
            nlb_stops=("operator", lambda x: (x == "NLB").sum()),
            total_routes=("routes_served", "sum"),
            unique_routes=("routes_served", lambda x: (x > 0).sum()),
            operator_diversity=("operator", "nunique"),
        )
        .reset_index()
    )

    # ------------------------------------------------------------------
    # 5. Merge with population and area
    # ------------------------------------------------------------------
    print("\n[5/6]  Merging with population data …")
    population = load_population()
    merged = district_stats.merge(population, on="district", how="outer", indicator=True)

    if (merged["_merge"] != "both").any():
        print("       ⚠ District name mismatches detected, applying normalised merge")
        merged = fuzzy_merge_districts(district_stats, population)
    else:
        merged = merged.drop(columns=["_merge"])

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    print("\n[6/6]  Saving processed dataset …")
    out_path = os.path.join(OUTPUT_DIR, "district_transport_data.csv")
    merged.to_csv(out_path, index=False)

    print("=" * 60)
    print(f"Processed data saved to: {os.path.abspath(out_path)}")
    print(f"Districts: {len(merged)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
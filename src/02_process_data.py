"""
02_process_data.py — Clean, merge, and spatially join the raw datasets
======================================================================

This is the second step of our pipeline.  It takes the raw files that
01_fetch_data.py downloaded and turns them into a single, analysis-
ready CSV with one row per Hong Kong district.

What it does
------------
1.  Parse KMB bus-stop JSON  →  DataFrame (stop_id, name, lat, long).
2.  Parse KMB route-stop JSON  →  count how many *unique routes* serve
    each stop.
3.  Parse tram-stop CSV  →  DataFrame (tram stop with lat/long).
4.  Load the GeoJSON district boundaries into a GeoDataFrame.
5.  **Spatial join** — assign each transport stop to the district whose
    polygon contains it, using GeoPandas' `sjoin`.
6.  Merge with population data and save.

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
import numpy as np
import pandas as pd
from shapely.geometry import Point

# Suppress the FutureWarning from geopandas about the legacy PyGEOS
# backend — it's noisy and not relevant to us.
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths (relative to the project root)
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR     = os.path.join(BASE_DIR, "data", "raw")
DATA_DIR    = os.path.join(BASE_DIR, "data")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output")


# ===================================================================
# Helper functions
# ===================================================================

def load_json(filename: str) -> dict:
    """Read a JSON file from the raw-data directory."""
    path = os.path.join(RAW_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_bus_stops(raw: dict) -> pd.DataFrame:
    """
    Convert the KMB bus-stop API response into a clean DataFrame.

    The API returns records like:
        {"stop": "AB12CD34...", "name_en": "...", "lat": "22.3...", "long": "114.1..."}

    We keep only the fields we need and cast coordinates to float.
    """
    records = raw.get("data", [])
    df = pd.DataFrame(records)

    # The API gives lat/long as strings — we need floats for geometry
    df["lat"]  = pd.to_numeric(df["lat"],  errors="coerce")
    df["long"] = pd.to_numeric(df["long"], errors="coerce")

    # Drop rows that somehow lack valid coordinates
    df = df.dropna(subset=["lat", "long"])

    # Rename so every dataset uses the same column names
    df = df.rename(columns={"stop": "stop_id", "name_en": "stop_name"})
    df["mode"] = "bus"   # tag the mode of transport

    return df[["stop_id", "stop_name", "lat", "long", "mode"]]


def count_routes_per_stop(raw_route_stops: dict) -> pd.DataFrame:
    """
    Figure out how many distinct routes pass through each bus stop.

    A stop served by 10 different bus routes is much more useful than
    one served by only 1 — this metric captures service breadth.
    """
    records = raw_route_stops.get("data", [])
    df = pd.DataFrame(records)

    # Each record has route + stop — we want unique routes per stop
    routes_per_stop = (
        df.groupby("stop")["route"]
        .nunique()
        .reset_index()
        .rename(columns={"stop": "stop_id", "route": "routes_served"})
    )
    return routes_per_stop


def parse_tram_stops(csv_path: str) -> pd.DataFrame:
    """
    Read the tram-stops CSV.

    The file has columns like StopCode, CombinedName, Latitude, Longitude.
    We standardise them to match the bus-stop format.
    """
    df = pd.read_csv(csv_path, encoding="utf-8")

    # The tram CSV column names vary between releases, so we
    # normalise them defensively.
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Try common column-name variants
    lat_col  = next((c for c in df.columns if "lat" in c), None)
    long_col = next((c for c in df.columns if "lon" in c or "lng" in c), None)
    name_col = next((c for c in df.columns if "name" in c or "combined" in c),
                     None)
    code_col = next((c for c in df.columns if "code" in c or "id" in c), None)

    if lat_col is None or long_col is None:
        print("  ⚠ Tram CSV doesn't have recognisable lat/long columns.")
        print(f"    Columns found: {list(df.columns)}")
        # Return an empty DataFrame so the pipeline can still continue
        return pd.DataFrame(columns=["stop_id", "stop_name",
                                     "lat", "long", "mode"])

    result = pd.DataFrame({
        "stop_id":   df[code_col].astype(str) if code_col else
                     [f"tram_{i}" for i in range(len(df))],
        "stop_name": df[name_col] if name_col else "Unknown Tram Stop",
        "lat":       pd.to_numeric(df[lat_col], errors="coerce"),
        "long":      pd.to_numeric(df[long_col], errors="coerce"),
        "mode":      "tram",
    })
    return result.dropna(subset=["lat", "long"])


def load_districts(geojson_path: str) -> gpd.GeoDataFrame:
    """
    Load the 18-district boundary GeoJSON into a GeoDataFrame.

    The official file uses EPSG:4326 (WGS 84), which is the same
    coordinate system as our lat/long data, so no reprojection needed.
    """
    gdf = gpd.read_file(geojson_path)

    # The district name field differs between file versions —
    # try common candidates.
    name_col = None
    for candidate in ["ENAME", "NAME_EN", "name_en", "DCNAME_EN",
                      "ename", "Name", "name", "DISTRICT"]:
        if candidate in gdf.columns:
            name_col = candidate
            break

    if name_col is None:
        # Last resort: look for any column whose values look like
        # English district names.
        for col in gdf.columns:
            sample = str(gdf[col].iloc[0])
            if any(d in sample for d in ["Central", "Wan Chai", "Sha Tin"]):
                name_col = col
                break

    if name_col is None:
        print("  ⚠ Could not find English district-name column.")
        print(f"    Columns available: {list(gdf.columns)}")
        sys.exit(1)

    gdf = gdf.rename(columns={name_col: "district"})

    # Keep only what we need (geometry + district name)
    return gdf[["district", "geometry"]]


def spatial_join_stops(stops_gdf: gpd.GeoDataFrame,
                       districts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Assign each transport stop to the district it falls within.

    This is the core GIS operation — a classic point-in-polygon test.
    GeoPandas handles the spatial index under the hood for performance.
    """
    joined = gpd.sjoin(stops_gdf, districts, how="left", predicate="within")
    return joined


def load_population() -> pd.DataFrame:
    """
    Load the curated population CSV.

    This gives us district population and area, which we need so we
    can compute per-capita metrics (stops per 10,000 people, etc.).
    """
    path = os.path.join(DATA_DIR, "population_by_district.csv")
    return pd.read_csv(path)


# ===================================================================
# Main pipeline
# ===================================================================

def main() -> None:
    print("=" * 60)
    print("02_process_data  ·  Cleaning and merging datasets")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load raw datasets
    # ------------------------------------------------------------------
    print("\n[1/6]  Loading bus stops …")
    bus_stops_raw = load_json("kmb_bus_stops.json")
    bus_stops = parse_bus_stops(bus_stops_raw)
    print(f"       → {len(bus_stops):,} bus stops parsed")

    print("\n[2/6]  Counting routes per stop …")
    route_stops_raw = load_json("kmb_route_stops.json")
    routes_per_stop = count_routes_per_stop(route_stops_raw)
    # Merge route counts back onto bus stops
    bus_stops = bus_stops.merge(routes_per_stop, on="stop_id", how="left")
    bus_stops["routes_served"] = bus_stops["routes_served"].fillna(0).astype(int)
    print(f"       → route counts attached")

    print("\n[3/6]  Loading tram stops …")
    tram_csv = os.path.join(RAW_DIR, "tram_stops.csv")
    tram_stops = parse_tram_stops(tram_csv)
    tram_stops["routes_served"] = 1    # tram is a single line each direction
    print(f"       → {len(tram_stops):,} tram stops parsed")

    # Combine all stops into a single DataFrame
    all_stops = pd.concat([bus_stops, tram_stops], ignore_index=True)
    print(f"\n       Total transport stops: {len(all_stops):,}")

    # ------------------------------------------------------------------
    # 2. Create point geometries and spatial join
    # ------------------------------------------------------------------
    print("\n[4/6]  Spatial join — assigning stops to districts …")
    geometry = [Point(xy) for xy in zip(all_stops["long"], all_stops["lat"])]
    stops_gdf = gpd.GeoDataFrame(all_stops, geometry=geometry, crs="EPSG:4326")

    districts = load_districts(
        os.path.join(RAW_DIR, "district_boundaries.json"))
    print(f"       → {len(districts)} district polygons loaded")

    joined = spatial_join_stops(stops_gdf, districts)

    # Some stops may fall outside all district polygons (e.g. stops on
    # the border or on reclaimed land).  We flag rather than drop them.
    n_unassigned = joined["district"].isna().sum()
    if n_unassigned > 0:
        print(f"       ⚠ {n_unassigned} stops fell outside all districts "
              "(border / reclaimed land — ignored in aggregation)")
    joined = joined.dropna(subset=["district"])

    # ------------------------------------------------------------------
    # 3. Aggregate per district
    # ------------------------------------------------------------------
    print("\n[5/6]  Aggregating metrics per district …")

    district_stats = (
        joined.groupby("district")
        .agg(
            total_stops=("stop_id", "count"),
            bus_stops=("mode", lambda x: (x == "bus").sum()),
            tram_stops=("mode", lambda x: (x == "tram").sum()),
            total_routes=("routes_served", "sum"),
            unique_routes=("routes_served", lambda x: (x > 0).sum()),
        )
        .reset_index()
    )

    # ------------------------------------------------------------------
    # 4. Merge with population data
    # ------------------------------------------------------------------
    print("\n[6/6]  Merging with population data …")
    population = load_population()

    # The district names might not match exactly (e.g. "Central and Western"
    # vs "Central & Western"), so we do a fuzzy-ish merge.
    # First try exact merge:
    merged = district_stats.merge(population, on="district", how="outer",
                                  indicator=True)

    # Check for unmatched districts so we can diagnose naming mismatches
    left_only  = merged[merged["_merge"] == "left_only"]["district"].tolist()
    right_only = merged[merged["_merge"] == "right_only"]["district"].tolist()

    if left_only or right_only:
        print(f"\n       ⚠ District name mismatches detected:")
        if left_only:
            print(f"         In GeoJSON but not population CSV: {left_only}")
        if right_only:
            print(f"         In population CSV but not GeoJSON: {right_only}")
        print("         Attempting fuzzy matching …")

        # Simple approach: try matching by stripping "&" vs "and", etc.
        merged = _fuzzy_merge_districts(district_stats, population)

    else:
        merged = merged.drop(columns=["_merge"])

    # ------------------------------------------------------------------
    # Save the result
    # ------------------------------------------------------------------
    out_path = os.path.join(OUTPUT_DIR, "district_transport_data.csv")
    merged.to_csv(out_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"Processed data saved to: {os.path.abspath(out_path)}")
    print(f"Districts: {len(merged)}")
    print(f"{'=' * 60}")


def _fuzzy_merge_districts(stats: pd.DataFrame,
                           pop: pd.DataFrame) -> pd.DataFrame:
    """
    Handle minor naming differences between the GeoJSON and our
    population CSV.

    For example, the GeoJSON might say "Central & Western" while we
    have "Central and Western".  This function normalises names before
    merging.
    """
    def normalise(name: str) -> str:
        """Lowercase, strip accents, unify '&' / 'and'."""
        n = name.strip().lower()
        n = n.replace("&", "and").replace("  ", " ")
        return n

    stats = stats.copy()
    pop   = pop.copy()

    stats["_key"] = stats["district"].apply(normalise)
    pop["_key"]   = pop["district"].apply(normalise)

    merged = stats.merge(pop, on="_key", how="outer",
                         suffixes=("_geo", "_pop"))

    # Prefer the population-CSV name (it's cleaner)
    merged["district"] = merged["district_pop"].fillna(merged["district_geo"])
    merged = merged.drop(columns=["district_geo", "district_pop", "_key"],
                         errors="ignore")

    return merged


if __name__ == "__main__":
    main()

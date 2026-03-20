"""
01_fetch_data.py — Download raw datasets from Hong Kong open-data APIs
=====================================================================

This script is the first stage of the pipeline. It fetches all raw
inputs needed by downstream scripts and writes them to data/raw/.

Data groups downloaded
----------------------
1. KMB bus datasets (stops, routes, route-stop mapping)
2. Citybus datasets (routes, route-stop mapping, stop details)
3. New Lantao Bus (NLB) datasets (routes, route-stop mapping, stop list)
4. District boundary GeoJSON
5. Topography sample points (elevation API)
6. Manmade ramp / step-free proxy points (Overpass / OSM)
7. Fine-grained population source metadata (WorldPop API)

All files are stored under data/raw/ so later scripts can run offline.

Author:  CCAI-9012 Group E
Date:    March 2026
"""

import json
import os
import sys
import time
from typing import Iterable

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration — keep all URLs centralised for maintainability.
# ---------------------------------------------------------------------------
URLS = {
    # KMB open API
    "kmb_bus_stops": "https://data.etabus.gov.hk/v1/transport/kmb/stop",
    "kmb_bus_routes": "https://data.etabus.gov.hk/v1/transport/kmb/route",
    "kmb_route_stop_map": "https://data.etabus.gov.hk/v1/transport/kmb/route-stop",

    # Citybus open API
    "citybus_routes": "https://rt.data.gov.hk/v2/transport/citybus/route/ctb",

    # New Lantao Bus open API
    "nlb_routes": "https://rt.data.gov.hk/v2/transport/nlb/route.php?action=list",

    # District boundaries (Home Affairs Department)
    "district_boundary": (
        "https://www.had.gov.hk/psi/"
        "hong-kong-administrative-boundaries/"
        "hksar_18_district_boundary.json"
    ),

    # Topography / elevation
    "elevation_api": "https://api.open-meteo.com/v1/elevation",

    # Barrier-free / ramp proxy features
    "overpass_api": "https://overpass-api.de/api/interpreter",

    # Fine-scale population source metadata
    "worldpop_metadata": "https://www.worldpop.org/rest/data/pop/wpgp?iso3=HKG",
}

# Where we save everything (relative to project root)
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def ensure_directory(path: str) -> None:
    """Create the directory (and parents) if it doesn't exist yet."""
    os.makedirs(path, exist_ok=True)


def download_json(url: str, label: str) -> dict:
    """
    Fetch a JSON endpoint and return the parsed dict.

    We keep a brief delay between calls for API friendliness.
    """
    print(f"  ↳ Downloading {label} …")
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    time.sleep(0.2)
    return response.json()


def download_file(url: str, dest_path: str, label: str) -> None:
    """Download any file and save it to disk using streamed chunks."""
    print(f"  ↳ Downloading {label} …")
    response = requests.get(url, timeout=120, stream=True)
    response.raise_for_status()

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    time.sleep(0.2)


def save_json(data: dict, path: str) -> None:
    """Write a Python dict to a nicely-formatted JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def chunked(items: list, size: int) -> Iterable[list]:
    """Yield fixed-size chunks from a list."""
    for i in range(0, len(items), size):
        yield items[i:i + size]


def frange(start: float, stop: float, step: float) -> Iterable[float]:
    """Simple float range helper."""
    v = start
    while v <= stop + 1e-12:
        yield v
        v += step


def fetch_citybus_data() -> tuple[dict, dict, dict]:
    """
    Fetch Citybus routes, route-stop links, and stop details.

    API pattern used:
    - route list: /route/ctb
    - route-stop: /route-stop/CTB/{route}/{direction}
    - stop detail: /stop/{stop_id}
    """
    routes = download_json(URLS["citybus_routes"], "Citybus routes")
    route_records = routes.get("data", [])

    route_stops = []
    for idx, r in enumerate(route_records, start=1):
        route = str(r.get("route", "")).strip()
        if not route:
            continue

        for direction in ["outbound", "inbound"]:
            endpoint = (
                "https://rt.data.gov.hk/v2/transport/citybus/"
                f"route-stop/CTB/{route}/{direction}"
            )
            try:
                payload = download_json(
                    endpoint,
                    f"Citybus route-stop {route} ({direction}) [{idx}/{len(route_records)}]",
                )
                for row in payload.get("data", []):
                    route_stops.append(
                        {
                            "co": "CTB",
                            "route": route,
                            "direction": direction,
                            "service_type": str(r.get("service_type", "1")),
                            "seq": row.get("seq"),
                            "stop": row.get("stop"),
                        }
                    )
            except requests.exceptions.RequestException:
                # Some route-direction combinations can be unavailable.
                continue

    # Hydrate stop details from unique stop IDs referenced in route-stop data.
    stop_ids = sorted({str(x.get("stop", "")) for x in route_stops if x.get("stop")})
    stops = []
    for i, stop_id in enumerate(stop_ids, start=1):
        endpoint = f"https://rt.data.gov.hk/v2/transport/citybus/stop/{stop_id}"
        try:
            payload = download_json(endpoint, f"Citybus stop detail {i}/{len(stop_ids)}")
            detail = payload.get("data", {})
            if detail:
                stops.append(detail)
        except requests.exceptions.RequestException:
            continue

    return routes, {"data": route_stops}, {"data": stops}


def fetch_nlb_data() -> tuple[dict, dict, dict]:
    """
    Fetch New Lantao Bus route list and stop lists per route.

    API pattern used:
    - route list: route.php?action=list
    - route stops: stop.php?action=list&routeId={routeId}
    """
    routes = download_json(URLS["nlb_routes"], "NLB routes")
    route_records = routes.get("routes", [])

    route_stops = []
    unique_stops = {}

    for idx, r in enumerate(route_records, start=1):
        route_id = r.get("routeId")
        if route_id is None:
            continue

        endpoint = (
            "https://rt.data.gov.hk/v2/transport/nlb/stop.php"
            f"?action=list&routeId={route_id}"
        )
        try:
            payload = download_json(
                endpoint,
                f"NLB stops for routeId={route_id} [{idx}/{len(route_records)}]",
            )
        except requests.exceptions.RequestException:
            continue

        stops = payload.get("stops", [])
        for seq, s in enumerate(stops, start=1):
            stop_id = str(s.get("stopId", "")).strip()
            if not stop_id:
                continue

            route_stops.append(
                {
                    "routeId": route_id,
                    "routeNo": r.get("routeNo"),
                    "stopId": stop_id,
                    "seq": seq,
                }
            )
            if stop_id not in unique_stops:
                unique_stops[stop_id] = s

    return (
        {"routes": route_records},
        {"data": route_stops},
        {"data": list(unique_stops.values())},
    )


def fetch_topography_points() -> pd.DataFrame:
    """
    Build a regular Hong Kong grid and request elevation values via API.

    We store sampled point elevations for later terrain-aware analysis.
    """
    print("  ↳ Downloading topography sample points (elevation API) …")

    # Approximate Hong Kong bounding box.
    lat_values = [round(v, 4) for v in frange(22.13, 22.57, 0.01)]
    lon_values = [round(v, 4) for v in frange(113.82, 114.51, 0.01)]
    points = [(lat, lon) for lat in lat_values for lon in lon_values]

    rows = []
    for part in chunked(points, size=90):
        lats = ",".join(str(p[0]) for p in part)
        lons = ",".join(str(p[1]) for p in part)
        params = {"latitude": lats, "longitude": lons}

        response = requests.get(URLS["elevation_api"], params=params, timeout=90)
        response.raise_for_status()
        payload = response.json()

        resp_lats = payload.get("latitude", [])
        resp_lons = payload.get("longitude", [])
        elevs = payload.get("elevation", [])
        for lat, lon, elev in zip(resp_lats, resp_lons, elevs):
            rows.append({"lat": lat, "lon": lon, "elevation_m": elev})

        time.sleep(0.2)

    return pd.DataFrame(rows)


def fetch_ramp_points() -> tuple[dict, pd.DataFrame]:
    """
    Fetch manmade ramp / step-free proxy points from OpenStreetMap.

    The query captures common tags related to ramps, lowered kerbs,
    and wheelchair-friendly paths.
    """
    print("  ↳ Downloading manmade ramp proxy data (Overpass API) …")
    overpass_query = """
    [out:json][timeout:180];
    (
      node["ramp"="yes"](22.13,113.82,22.57,114.51);
      node["kerb"~"lowered|flush"](22.13,113.82,22.57,114.51);
      node["wheelchair"="yes"](22.13,113.82,22.57,114.51);
      way["ramp"="yes"](22.13,113.82,22.57,114.51);
      way["wheelchair"="yes"](22.13,113.82,22.57,114.51);
    );
    out center tags;
    """.strip()

    response = requests.post(
        URLS["overpass_api"],
        data=overpass_query.encode("utf-8"),
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()

    rows = []
    for el in payload.get("elements", []):
        if "lat" in el and "lon" in el:
            lat, lon = el["lat"], el["lon"]
        elif "center" in el:
            lat, lon = el["center"].get("lat"), el["center"].get("lon")
        else:
            continue

        tags = el.get("tags", {})
        rows.append(
            {
                "osm_id": el.get("id"),
                "osm_type": el.get("type"),
                "lat": lat,
                "lon": lon,
                "ramp": tags.get("ramp"),
                "kerb": tags.get("kerb"),
                "wheelchair": tags.get("wheelchair"),
                "highway": tags.get("highway"),
            }
        )

    return payload, pd.DataFrame(rows)


def fetch_worldpop_metadata() -> dict:
    """
    Fetch metadata for fine-resolution population sources (WorldPop).

    The metadata is stored directly and can be used later to select
    suitable high-resolution gridded population products.
    """
    return download_json(URLS["worldpop_metadata"], "WorldPop metadata (Hong Kong)")


def main() -> None:
    """Entry point — download all data dependencies for the pipeline."""
    print("=" * 60)
    print("01_fetch_data  ·  Downloading Hong Kong transport datasets")
    print("=" * 60)

    ensure_directory(RAW_DIR)

    # ------------------------------------------------------------------
    # 1. KMB datasets
    # ------------------------------------------------------------------
    kmb_stops = download_json(URLS["kmb_bus_stops"], "KMB bus-stop locations")
    save_json(kmb_stops, os.path.join(RAW_DIR, "kmb_bus_stops.json"))
    print(f"    ✓ {len(kmb_stops.get('data', [])):,} KMB stops saved\n")

    kmb_routes = download_json(URLS["kmb_bus_routes"], "KMB bus routes")
    save_json(kmb_routes, os.path.join(RAW_DIR, "kmb_bus_routes.json"))
    print(f"    ✓ {len(kmb_routes.get('data', [])):,} KMB route records saved\n")

    kmb_route_stops = download_json(URLS["kmb_route_stop_map"], "KMB route-stop mapping")
    save_json(kmb_route_stops, os.path.join(RAW_DIR, "kmb_route_stops.json"))
    print(f"    ✓ {len(kmb_route_stops.get('data', [])):,} KMB route-stop links saved\n")

    # ------------------------------------------------------------------
    # 2. Citybus datasets
    # ------------------------------------------------------------------
    city_routes, city_route_stops, city_stops = fetch_citybus_data()
    save_json(city_routes, os.path.join(RAW_DIR, "citybus_routes.json"))
    save_json(city_route_stops, os.path.join(RAW_DIR, "citybus_route_stops.json"))
    save_json(city_stops, os.path.join(RAW_DIR, "citybus_stops.json"))
    print(
        "    ✓ Citybus saved "
        f"({len(city_routes.get('data', [])):,} routes, "
        f"{len(city_route_stops.get('data', [])):,} route-stop links, "
        f"{len(city_stops.get('data', [])):,} stops)\n"
    )

    # ------------------------------------------------------------------
    # 3. NLB datasets
    # ------------------------------------------------------------------
    nlb_routes, nlb_route_stops, nlb_stops = fetch_nlb_data()
    save_json(nlb_routes, os.path.join(RAW_DIR, "nlb_routes.json"))
    save_json(nlb_route_stops, os.path.join(RAW_DIR, "nlb_route_stops.json"))
    save_json(nlb_stops, os.path.join(RAW_DIR, "nlb_stops.json"))
    print(
        "    ✓ NLB saved "
        f"({len(nlb_routes.get('routes', [])):,} routes, "
        f"{len(nlb_route_stops.get('data', [])):,} route-stop links, "
        f"{len(nlb_stops.get('data', [])):,} stops)\n"
    )

    # ------------------------------------------------------------------
    # 4. District boundaries
    # ------------------------------------------------------------------
    boundary_path = os.path.join(RAW_DIR, "district_boundaries.json")
    download_file(URLS["district_boundary"], boundary_path, "District boundary GeoJSON")
    print("    ✓ District boundaries saved\n")

    # ------------------------------------------------------------------
    # 5. Topography points
    # ------------------------------------------------------------------
    topo_df = fetch_topography_points()
    topo_path = os.path.join(RAW_DIR, "hk_topography_points.csv")
    topo_df.to_csv(topo_path, index=False)
    print(f"    ✓ Topography points saved ({len(topo_df):,} samples)\n")

    # ------------------------------------------------------------------
    # 6. Ramp proxy points
    # ------------------------------------------------------------------
    ramps_raw, ramps_df = fetch_ramp_points()
    save_json(ramps_raw, os.path.join(RAW_DIR, "hk_ramps_overpass.json"))
    ramps_df.to_csv(os.path.join(RAW_DIR, "hk_ramps_points.csv"), index=False)
    print(f"    ✓ Ramp proxy points saved ({len(ramps_df):,} features)\n")

    # ------------------------------------------------------------------
    # 7. Fine-population source metadata
    # ------------------------------------------------------------------
    worldpop_metadata = fetch_worldpop_metadata()
    save_json(worldpop_metadata, os.path.join(RAW_DIR, "worldpop_hkg_metadata.json"))

    # Best-effort TIFF download if metadata includes raster URL(s).
    tiff_urls = []

    def collect_tiff_urls(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                collect_tiff_urls(v)
        elif isinstance(obj, list):
            for v in obj:
                collect_tiff_urls(v)
        elif isinstance(obj, str):
            lo = obj.lower()
            if lo.endswith(".tif") or lo.endswith(".tiff"):
                tiff_urls.append(obj)

    collect_tiff_urls(worldpop_metadata)

    if tiff_urls:
        pop_raster_path = os.path.join(RAW_DIR, "worldpop_hkg.tif")
        try:
            download_file(tiff_urls[0], pop_raster_path, "WorldPop fine population raster")
            print("    ✓ WorldPop raster saved\n")
        except requests.exceptions.RequestException:
            print("    ⚠ WorldPop raster URL found but download failed; metadata still saved\n")
    else:
        print("    ⚠ No TIFF URL found in WorldPop metadata; metadata still saved\n")

    print("=" * 60)
    print("All datasets downloaded successfully.")
    print(f"Raw data saved to: {os.path.abspath(RAW_DIR)}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.RequestException as exc:
        print(f"\nDownload failed: {exc}", file=sys.stderr)
        print("Check your internet connection and try again.", file=sys.stderr)
        sys.exit(1)
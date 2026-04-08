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
5. Hong Kong road network graph + edges overlay (OpenStreetMap)
6. Topography sample points (official HK DTM 5m)
7. Manmade ramp / step-free proxy points (Overpass / OSM)
8. Fine-grained population source metadata (WorldPop API)

All files are stored under data/raw/ so later scripts can run offline.

Author:  CCAI-9012 Group E
Date:    April 2026
"""

import json
import os
import sys
import time
import argparse
import zipfile
from typing import Iterable

import geopandas as gpd
import osmnx as ox
import pandas as pd
import requests
from pyproj import Transformer

# Ensure Unicode status logs render on Windows terminals.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

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

    # Topography / terrain (official HK DTM 5m)
    "hk_dtm_5m_zip": (
        "https://res.data.gov.hk/api/get-download-file"
        "?name=https%3A%2F%2Fwww.landsd.gov.hk%2Flandsd_psi_data%2FSMO%2Fdata%2FWhole_HK_DTM_5m.zip"
    ),

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


def file_ready(path: str) -> bool:
    """Return True when a file exists and is non-empty."""
    return os.path.exists(path) and os.path.getsize(path) > 2


def load_json(path: str) -> dict:
    """Load JSON from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_cached_json(path: str, label: str) -> dict | None:
    """Try to load cached JSON; return None if unavailable/unreadable."""
    if not file_ready(path):
        return None
    try:
        payload = load_json(path)
        print(f"  ↳ Using cached {label}")
        return payload
    except (json.JSONDecodeError, OSError):
        print(f"  ↳ Cached {label} is unreadable; re-downloading …")
        return None


def maybe_download_json(url: str, path: str, label: str, refresh: bool) -> dict:
    """Use cached JSON if present unless refresh is requested."""
    if not refresh:
        cached = load_cached_json(path, label)
        if cached is not None:
            return cached

    payload = download_json(url, label)
    save_json(payload, path)
    return payload


def maybe_download_file(url: str, dest_path: str, label: str, refresh: bool) -> None:
    """Use cached file if present unless refresh is requested."""
    if not refresh and file_ready(dest_path):
        print(f"  ↳ Using cached {label}")
        return
    download_file(url, dest_path, label)


def download_json(url: str, label: str) -> dict:
    """
    Fetch a JSON endpoint and return the parsed dict.

    We keep a brief delay between calls for API friendliness.
    """
    print(f"  ↳ Downloading {label} …")
    return request_json_with_retry(
        url=url,
        label=label,
        timeout=90,
    )


def request_json_with_retry(
    url: str,
    label: str,
    params: dict | None = None,
    timeout: int = 90,
    max_retries: int = 6,
    base_sleep: float = 1.5,
) -> dict:
    """
    GET JSON endpoint with retry/backoff, including explicit 429 handling.

    If server sends Retry-After, that delay is respected.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                try:
                    wait_s = float(retry_after) if retry_after is not None else 0.0
                except ValueError:
                    wait_s = 0.0
                if wait_s <= 0:
                    wait_s = min(120.0, base_sleep * (2 ** (attempt - 1)))

                if attempt == max_retries:
                    response.raise_for_status()

                print(
                    f"    ⚠ Rate-limited while fetching {label}; "
                    f"retrying in {wait_s:.1f}s (attempt {attempt}/{max_retries})"
                )
                time.sleep(wait_s)
                continue

            response.raise_for_status()
            time.sleep(0.2)
            return response.json()

        except requests.exceptions.RequestException as exc:
            if attempt == max_retries:
                raise
            wait_s = min(120.0, base_sleep * (2 ** (attempt - 1)))
            print(
                f"    ⚠ Temporary error fetching {label}: {exc}; "
                f"retrying in {wait_s:.1f}s (attempt {attempt}/{max_retries})"
            )
            time.sleep(wait_s)

    raise RuntimeError(f"Unexpected retry loop termination for {label}")


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


def fetch_topography_points(refresh: bool = False) -> pd.DataFrame:
    """
    Build topography points by sampling the official HK 5m DTM ASCII grid.

    The DTM grid is in HK1980 Grid (EPSG:2326). We sample at a coarser
    stride for efficiency, then convert sampled points to WGS84 lat/lon
    for downstream distance calculations.
    """
    print("  ↳ Building topography points from official HK DTM (5m) …")

    dtm_zip_path = os.path.join(RAW_DIR, "hk_dtm_5m.zip")
    dtm_extract_dir = os.path.join(RAW_DIR, "hk_dtm_5m")

    maybe_download_file(
        URLS["hk_dtm_5m_zip"],
        dtm_zip_path,
        "HK DTM 5m ZIP",
        refresh=refresh,
    )

    ensure_directory(dtm_extract_dir)
    asc_path = None
    with zipfile.ZipFile(dtm_zip_path, "r") as zf:
        asc_members = [n for n in zf.namelist() if n.lower().endswith(".asc")]
        if not asc_members:
            raise ValueError("No .asc file found in HK DTM ZIP.")

        asc_member = asc_members[0]
        asc_name = os.path.basename(asc_member)
        asc_path = os.path.join(dtm_extract_dir, asc_name)
        if not file_ready(asc_path):
            print("  ↳ Extracting DTM ASCII grid …")
            zf.extract(asc_member, dtm_extract_dir)
            extracted_path = os.path.join(dtm_extract_dir, asc_member)
            if extracted_path != asc_path:
                os.replace(extracted_path, asc_path)

    # 5m grid sampled every 40 cells (~200m) to keep runtime/file size practical.
    sample_stride_cells = 40

    transformer = Transformer.from_crs("EPSG:2326", "EPSG:4326", always_xy=True)
    rows: list[dict] = []

    print("  ↳ Sampling DTM grid and converting HK1980 -> WGS84 …")
    with open(asc_path, "r", encoding="utf-8", errors="ignore") as f:
        header = {}
        for _ in range(6):
            key, value = f.readline().strip().split(maxsplit=1)
            header[key.lower()] = value

        ncols = int(float(header["ncols"]))
        nrows = int(float(header["nrows"]))
        xll = float(header.get("xllcorner", header.get("xllcenter")))
        yll = float(header.get("yllcorner", header.get("yllcenter")))
        cellsize = float(header["cellsize"])
        nodata = float(header.get("nodata_value", "-9999"))

        sample_cols = list(range(0, ncols, sample_stride_cells))

        for row_idx in range(nrows):
            line = f.readline()
            if not line:
                break

            if row_idx % sample_stride_cells != 0:
                continue

            vals = line.strip().split()
            if len(vals) < ncols:
                continue

            y = yll + (nrows - row_idx - 0.5) * cellsize
            xs = []
            zs = []
            for col_idx in sample_cols:
                try:
                    z = float(vals[col_idx])
                except (ValueError, IndexError):
                    continue
                if z == nodata:
                    continue

                x = xll + (col_idx + 0.5) * cellsize
                xs.append(x)
                zs.append(z)

            if not xs:
                continue

            ys = [y] * len(xs)
            lons, lats = transformer.transform(xs, ys)
            rows.extend(
                {
                    "lat": float(lat),
                    "lon": float(lon),
                    "elevation_m": float(z),
                }
                for lat, lon, z in zip(lats, lons, zs)
            )

    if not rows:
        raise ValueError("DTM sampling produced 0 valid topography points.")

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


def fetch_hk_road_graph(
    district_boundary_path: str,
    graphml_path: str,
    edges_geojson_path: str,
    refresh: bool,
) -> None:
    """
    Build and persist a Hong Kong road network graph from OpenStreetMap.

    We use district polygons as the exact boundary mask and download the
    drivable road network via OSMnx. Outputs are cached in data/raw/.
    """
    if not refresh and file_ready(graphml_path) and file_ready(edges_geojson_path):
        print("  ↳ Using cached Hong Kong road graph")
        return

    print("  ↳ Building Hong Kong road graph from OpenStreetMap (OSMnx) …")

    districts = gpd.read_file(district_boundary_path)
    districts = districts.to_crs("EPSG:4326")
    hk_polygon = districts.geometry.unary_union

    graph = ox.graph_from_polygon(
        hk_polygon,
        network_type="drive",
        simplify=True,
        retain_all=True,
        truncate_by_edge=True,
    )

    # Persist raw graph for shortest-path modelling.
    ox.save_graphml(graph, graphml_path)

    # Save edge geometry for overlays and QA checks.
    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True)
    edges.to_file(edges_geojson_path, driver="GeoJSON")

    print(
        "    ✓ Road graph saved "
        f"({graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges)\n"
    )


def main(refresh: bool = False) -> None:
    """Entry point — download all data dependencies for the pipeline."""
    print("=" * 60)
    print("01_fetch_data  ·  Downloading Hong Kong transport datasets")
    print("=" * 60)

    if refresh:
        print("Refresh mode: force re-download of all datasets\n")
    else:
        print("Cache mode: reuse existing files in data/raw when available\n")

    ensure_directory(RAW_DIR)

    # ------------------------------------------------------------------
    # 1. KMB datasets
    # ------------------------------------------------------------------
    kmb_stops_path = os.path.join(RAW_DIR, "kmb_bus_stops.json")
    kmb_stops = maybe_download_json(
        URLS["kmb_bus_stops"],
        kmb_stops_path,
        "KMB bus-stop locations",
        refresh,
    )
    print(f"    ✓ {len(kmb_stops.get('data', [])):,} KMB stops saved\n")

    kmb_routes_path = os.path.join(RAW_DIR, "kmb_bus_routes.json")
    kmb_routes = maybe_download_json(
        URLS["kmb_bus_routes"],
        kmb_routes_path,
        "KMB bus routes",
        refresh,
    )
    print(f"    ✓ {len(kmb_routes.get('data', [])):,} KMB route records saved\n")

    kmb_route_stops_path = os.path.join(RAW_DIR, "kmb_route_stops.json")
    kmb_route_stops = maybe_download_json(
        URLS["kmb_route_stop_map"],
        kmb_route_stops_path,
        "KMB route-stop mapping",
        refresh,
    )
    print(f"    ✓ {len(kmb_route_stops.get('data', [])):,} KMB route-stop links saved\n")

    # ------------------------------------------------------------------
    # 2. Citybus datasets
    # ------------------------------------------------------------------
    city_routes_path = os.path.join(RAW_DIR, "citybus_routes.json")
    city_route_stops_path = os.path.join(RAW_DIR, "citybus_route_stops.json")
    city_stops_path = os.path.join(RAW_DIR, "citybus_stops.json")
    if not refresh and all(file_ready(p) for p in [city_routes_path, city_route_stops_path, city_stops_path]):
        city_routes = load_json(city_routes_path)
        city_route_stops = load_json(city_route_stops_path)
        city_stops = load_json(city_stops_path)
        print("  ↳ Using cached Citybus datasets")
    else:
        city_routes, city_route_stops, city_stops = fetch_citybus_data()
        save_json(city_routes, city_routes_path)
        save_json(city_route_stops, city_route_stops_path)
        save_json(city_stops, city_stops_path)
    print(
        "    ✓ Citybus saved "
        f"({len(city_routes.get('data', [])):,} routes, "
        f"{len(city_route_stops.get('data', [])):,} route-stop links, "
        f"{len(city_stops.get('data', [])):,} stops)\n"
    )

    # ------------------------------------------------------------------
    # 3. NLB datasets
    # ------------------------------------------------------------------
    nlb_routes_path = os.path.join(RAW_DIR, "nlb_routes.json")
    nlb_route_stops_path = os.path.join(RAW_DIR, "nlb_route_stops.json")
    nlb_stops_path = os.path.join(RAW_DIR, "nlb_stops.json")
    if not refresh and all(file_ready(p) for p in [nlb_routes_path, nlb_route_stops_path, nlb_stops_path]):
        nlb_routes = load_json(nlb_routes_path)
        nlb_route_stops = load_json(nlb_route_stops_path)
        nlb_stops = load_json(nlb_stops_path)
        print("  ↳ Using cached NLB datasets")
    else:
        nlb_routes, nlb_route_stops, nlb_stops = fetch_nlb_data()
        save_json(nlb_routes, nlb_routes_path)
        save_json(nlb_route_stops, nlb_route_stops_path)
        save_json(nlb_stops, nlb_stops_path)
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
    maybe_download_file(URLS["district_boundary"], boundary_path, "District boundary GeoJSON", refresh)
    print("    ✓ District boundaries saved\n")

    # ------------------------------------------------------------------
    # 5. Hong Kong road network graph (OSM)
    # ------------------------------------------------------------------
    road_graphml_path = os.path.join(RAW_DIR, "hk_roads_drive.graphml")
    road_edges_geojson_path = os.path.join(RAW_DIR, "hk_roads_edges.geojson")
    fetch_hk_road_graph(
        district_boundary_path=boundary_path,
        graphml_path=road_graphml_path,
        edges_geojson_path=road_edges_geojson_path,
        refresh=refresh,
    )

    # ------------------------------------------------------------------
    # 6. Topography points
    # ------------------------------------------------------------------
    topo_path = os.path.join(RAW_DIR, "hk_topography_points.csv")
    use_cached_topo = False
    if not refresh and file_ready(topo_path):
        try:
            cached_topo_df = pd.read_csv(topo_path)
            required_cols = {"lat", "lon", "elevation_m"}
            if (not cached_topo_df.empty) and required_cols.issubset(cached_topo_df.columns):
                topo_df = cached_topo_df
                use_cached_topo = True
                print("  ↳ Using cached topography sample points")
            else:
                print("  ↳ Cached topography is empty/invalid; re-downloading …")
        except (pd.errors.EmptyDataError, OSError):
            print("  ↳ Cached topography unreadable; re-downloading …")

    if not use_cached_topo:
        topo_df = fetch_topography_points(refresh=refresh)
        if topo_df.empty:
            raise requests.exceptions.RequestException(
                "Topography download returned empty dataset; file not written."
            )
        topo_df.to_csv(topo_path, index=False)
    print(f"    ✓ Topography points saved ({len(topo_df):,} samples)\n")

    # ------------------------------------------------------------------
    # 7. Ramp proxy points
    # ------------------------------------------------------------------
    ramps_raw_path = os.path.join(RAW_DIR, "hk_ramps_overpass.json")
    ramps_csv_path = os.path.join(RAW_DIR, "hk_ramps_points.csv")
    if not refresh and all(file_ready(p) for p in [ramps_raw_path, ramps_csv_path]):
        ramps_raw = load_json(ramps_raw_path)
        ramps_df = pd.read_csv(ramps_csv_path)
        print("  ↳ Using cached ramp proxy datasets")
    else:
        ramps_raw, ramps_df = fetch_ramp_points()
        save_json(ramps_raw, ramps_raw_path)
        ramps_df.to_csv(ramps_csv_path, index=False)
    print(f"    ✓ Ramp proxy points saved ({len(ramps_df):,} features)\n")

    # ------------------------------------------------------------------
    # 8. Fine-population source metadata
    # ------------------------------------------------------------------
    worldpop_meta_path = os.path.join(RAW_DIR, "worldpop_hkg_metadata.json")
    if not refresh and file_ready(worldpop_meta_path):
        worldpop_metadata = load_json(worldpop_meta_path)
        print("  ↳ Using cached WorldPop metadata (Hong Kong)")
    else:
        worldpop_metadata = fetch_worldpop_metadata()
        save_json(worldpop_metadata, worldpop_meta_path)

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
        if not refresh and file_ready(pop_raster_path):
            print("  ↳ Using cached WorldPop fine population raster")
            print("    ✓ WorldPop raster saved\n")
        else:
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
        parser = argparse.ArgumentParser(
            description="Fetch Hong Kong transport datasets with cache-aware reuse.",
        )
        parser.add_argument(
            "--refresh",
            action="store_true",
            help="Force re-download even if cached files already exist.",
        )
        args = parser.parse_args()

        main(refresh=args.refresh)
    except requests.exceptions.RequestException as exc:
        print(f"\nDownload failed: {exc}", file=sys.stderr)
        print("Check your internet connection and try again.", file=sys.stderr)
        sys.exit(1)
"""
01_fetch_data.py — Download raw datasets from Hong Kong open-data APIs
=====================================================================

This is the first script in our AI pipeline. It grabs four categories
of data that we need for the transport-inequality analysis:

    1. KMB bus-stop locations   (JSON API)
    2. KMB bus routes           (JSON API)
    3. KMB route-stop mapping   (JSON API)
    4. Hong Kong tram stops     (CSV download)
    5. District boundary map    (GeoJSON download)

All files are saved into  data/raw/  so the rest of the pipeline can
work offline once this script has run at least once.

Data sources
------------
- KMB (Kowloon Motor Bus)  : https://data.etabus.gov.hk
- Tram stops               : https://data.gov.hk  (Hong Kong Tramways)
- District boundaries      : https://www.had.gov.hk  (Home Affairs Dept)

Author:  CCAI-9012 Group E
Date:    March 2026
"""

import json
import os
import sys
import time

import requests

# ---------------------------------------------------------------------------
# Configuration — all our download URLs in one place so they're easy to
# update if the government changes its API endpoints.
# ---------------------------------------------------------------------------
URLS = {
    # KMB real-time ETA API (public, no key needed)
    "bus_stops":       "https://data.etabus.gov.hk/v1/transport/kmb/stop",
    "bus_routes":      "https://data.etabus.gov.hk/v1/transport/kmb/route",
    "route_stop_map":  "https://data.etabus.gov.hk/v1/transport/kmb/route-stop",

    # Tramways open data — a simple CSV hosted on data.gov.hk
    "tram_stops":      ("https://static.data.gov.hk/tramways/datasets/"
                        "tram_stops/summary_tram_stops_en.csv"),

    # Home Affairs Department — GeoJSON polygon boundaries for 18 districts
    "district_boundary": ("https://www.had.gov.hk/psi/"
                          "hong-kong-administrative-boundaries/"
                          "hksar_18_district_boundary.json"),
}

# Where we save everything (relative to project root)
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def ensure_directory(path: str) -> None:
    """Create the directory (and parents) if it doesn't exist yet."""
    os.makedirs(path, exist_ok=True)


def download_json(url: str, label: str) -> dict:
    """
    Fetch a JSON endpoint and return the parsed dict.

    We add a small delay between calls to be polite to the server —
    these are public APIs and we don't want to hammer them.
    """
    print(f"  ↳ Downloading {label} …")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    time.sleep(0.5)          # polite pause between requests
    return response.json()


def download_file(url: str, dest_path: str, label: str) -> None:
    """
    Download any file (CSV, GeoJSON, etc.) and write it to disk.

    We stream the response so we can handle larger files without
    blowing up memory, although these datasets are pretty small.
    """
    print(f"  ↳ Downloading {label} …")
    response = requests.get(url, timeout=60, stream=True)
    response.raise_for_status()

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    time.sleep(0.5)


def save_json(data: dict, path: str) -> None:
    """Write a Python dict to a nicely-formatted JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    """
    Entry point — download every dataset and save it locally.

    The script is deliberately sequential (not async) because we're
    only hitting a handful of endpoints and simplicity matters more
    than speed for a research project.
    """
    print("=" * 60)
    print("01_fetch_data  ·  Downloading Hong Kong transport datasets")
    print("=" * 60)

    ensure_directory(RAW_DIR)

    # ------------------------------------------------------------------
    # 1. KMB bus stops — every franchised bus stop in Hong Kong
    # ------------------------------------------------------------------
    bus_stops = download_json(URLS["bus_stops"], "KMB bus-stop locations")
    save_json(bus_stops, os.path.join(RAW_DIR, "kmb_bus_stops.json"))
    n_stops = len(bus_stops.get("data", []))
    print(f"    ✓ {n_stops:,} bus stops saved\n")

    # ------------------------------------------------------------------
    # 2. KMB bus routes — which routes exist (number, direction, etc.)
    # ------------------------------------------------------------------
    bus_routes = download_json(URLS["bus_routes"], "KMB bus routes")
    save_json(bus_routes, os.path.join(RAW_DIR, "kmb_bus_routes.json"))
    n_routes = len(bus_routes.get("data", []))
    print(f"    ✓ {n_routes:,} route records saved\n")

    # ------------------------------------------------------------------
    # 3. KMB route ↔ stop mapping — tells us which stops serve which
    #    routes, so we can count how many routes pass through each area.
    # ------------------------------------------------------------------
    route_stops = download_json(URLS["route_stop_map"],
                                "KMB route-stop mapping")
    save_json(route_stops, os.path.join(RAW_DIR, "kmb_route_stops.json"))
    n_rs = len(route_stops.get("data", []))
    print(f"    ✓ {n_rs:,} route–stop links saved\n")

    # ------------------------------------------------------------------
    # 4. Tram stops — the iconic Hong Kong Island tram network
    #    This is a simple CSV, so we just download the file directly.
    # ------------------------------------------------------------------
    tram_path = os.path.join(RAW_DIR, "tram_stops.csv")
    download_file(URLS["tram_stops"], tram_path, "Tram stop locations")
    print("    ✓ Tram stops CSV saved\n")

    # ------------------------------------------------------------------
    # 5. District boundaries — GeoJSON polygons for all 18 districts.
    #    We need this to figure out which district each stop falls in.
    # ------------------------------------------------------------------
    boundary_path = os.path.join(RAW_DIR, "district_boundaries.json")
    download_file(URLS["district_boundary"], boundary_path,
                  "District boundary GeoJSON")
    print("    ✓ District boundaries saved\n")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print("All datasets downloaded successfully.")
    print(f"Raw data saved to: {os.path.abspath(RAW_DIR)}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.RequestException as exc:
        # If a download fails (e.g. network issues), we print a clear
        # error instead of a scary traceback.
        print(f"\n✘ Download failed: {exc}", file=sys.stderr)
        print("  Check your internet connection and try again.",
              file=sys.stderr)
        sys.exit(1)

# Datasets

This directory contains the curated datasets used by the project.
Raw API downloads are stored in `data/raw/` (gitignored — re-downloadable via `src/01_fetch_data.py`).

## Dataset Catalogue

| # | Dataset | Source | Licence | Notes |
|---|---------|--------|---------|-------|
| 1 | **KMB Bus Stops** | [KMB ETA API](https://data.etabus.gov.hk/v1/transport/kmb/stop) | HK Gov Open Data | ~3,900 stops with lat/long |
| 2 | **KMB Bus Routes** | [KMB ETA API](https://data.etabus.gov.hk/v1/transport/kmb/route) | HK Gov Open Data | All franchised KMB routes |
| 3 | **KMB Route-Stop Mapping** | [KMB ETA API](https://data.etabus.gov.hk/v1/transport/kmb/route-stop) | HK Gov Open Data | Links routes → stops |
| 4 | **Tram Stops** | [DATA.GOV.HK](https://static.data.gov.hk/tramways/datasets/tram_stops/summary_tram_stops_en.csv) | HK Gov Open Data | ~120 tram stops |
| 5 | **District Boundaries** | [Home Affairs Dept](https://www.had.gov.hk/psi/hong-kong-administrative-boundaries/hksar_18_district_boundary.json) | HK Gov Open Data | GeoJSON polygons for 18 districts |
| 6 | **Population by District** | [Census & Statistics Dept](https://www.censtatd.gov.hk/) | HK Gov Open Data | Curated from 2021 Census data |

## File: `population_by_district.csv`

Manually compiled from Census & Statistics Department Table 110-02001.  
Contains district name, mid-2021 population estimate, and land area in km².

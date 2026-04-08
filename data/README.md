# Data Documentation

This folder contains curated controls and all reproducible raw data assets used by the Hong Kong transport accessibility pipeline.

The ingestion script (`src/01_fetch_data.py`) populates `data/raw/` so downstream scripts can run offline once data is fetched.

## Directory Contents

- `population_by_district.csv`
  - district-level population and area controls
- `raw/`
  - downloaded API payloads
  - generated geospatial derivatives
  - road graph files

## Data Layers and Role

The data stack supports five model layers:

1. Service supply layer
   - bus stops, routes, route-stop mappings
2. Spatial context layer
   - district polygons
3. Network layer
   - Hong Kong road graph for shortest-path modelling
4. Physical accessibility layer
   - topography + ramp proxies
5. Population layer
   - district controls + optional fine population metadata

## Dataset Catalogue

| # | Dataset | Local Output | Source | Typical Use |
|---|---------|--------------|--------|-------------|
| 1 | KMB Bus Stops | `raw/kmb_bus_stops.json` | KMB Open Data API | Stop geometry and IDs |
| 2 | KMB Bus Routes | `raw/kmb_bus_routes.json` | KMB Open Data API | Route inventory |
| 3 | KMB Route-Stop Mapping | `raw/kmb_route_stops.json` | KMB Open Data API | Route service per stop |
| 4 | Citybus Routes | `raw/citybus_routes.json` | Citybus Open Data API | Route inventory |
| 5 | Citybus Route-Stop Mapping | `raw/citybus_route_stops.json` | Citybus Open Data API | Route direction links |
| 6 | Citybus Stops | `raw/citybus_stops.json` | Citybus Open Data API | Stop geometry and IDs |
| 7 | NLB Routes | `raw/nlb_routes.json` | NLB Open Data API | Route inventory |
| 8 | NLB Route-Stop Mapping | `raw/nlb_route_stops.json` | NLB Open Data API | Route service per stop |
| 9 | NLB Stops | `raw/nlb_stops.json` | NLB Open Data API | Stop geometry and IDs |
| 10 | District Boundaries | `raw/district_boundaries.json` | HAD Open Data | District polygon assignment |
| 11 | HK Road Graph (GraphML) | `raw/hk_roads_drive.graphml` | OpenStreetMap via OSMnx | Road shortest-path distance model |
| 12 | HK Road Edges (GeoJSON) | `raw/hk_roads_edges.geojson` | OpenStreetMap via OSMnx | Map overlay and QA |
| 13 | Topography Sample Points | `raw/hk_topography_points.csv` | HK DTM 5m | Terrain ruggedness and visuals |
| 14 | Ramp Proxy Raw | `raw/hk_ramps_overpass.json` | Overpass / OSM | Extraction traceability |
| 15 | Ramp Proxy Points | `raw/hk_ramps_points.csv` | Overpass / OSM | Ramp proximity metrics |
| 16 | Population by District | `population_by_district.csv` | Curated district controls | Per-capita normalization |
| 17 | WorldPop Metadata | `raw/worldpop_hkg_metadata.json` | WorldPop API | Fine population source references |
| 18 | WorldPop Raster (optional) | `raw/worldpop_hkg.tif` | WorldPop endpoint | Optional high-resolution context |

## Core Schema Notes

### `population_by_district.csv`

Required columns:
- `district`
- `population`
- `area_km2`

Used for:
- district-normalized indicators
- micro-grid population redistribution anchor totals

### `raw/hk_roads_drive.graphml`

Stored network object with:
- road nodes containing geographic coordinates (`x`, `y`)
- road edges containing weighted `length` attributes (meters)

Used for:
- nearest-stop road distance in accessibility scoring
- road-aware spacing in simulation candidate selection

### `raw/hk_roads_edges.geojson`

Edge geometries for visualization and QA.

Used for:
- road overlay in interactive maps
- manual inspection of network coverage

### `raw/hk_topography_points.csv`

Expected columns:
- `lat`
- `lon`
- `elevation_m`

Used for:
- terrain ruggedness penalties
- 3D topography visualization

### `raw/hk_ramps_points.csv`

Expected columns:
- `lat`
- `lon`
- optional OSM tag metadata

Used for:
- nearest-ramp proximity factors
- district ramp coverage metrics

## Refresh Workflow

Recommended refresh process:
1. Run `python src/01_fetch_data.py`
2. Confirm expected files in `data/raw/`
3. Re-run full pipeline to avoid mixed-version outputs
4. Archive timestamped snapshot of `data/raw/` for reproducibility

## Data Quality Notes

- API payloads can vary over time (schema or content drift).
- OSM-derived layers depend on community map completeness.
- District boundaries include maritime extents; remote policy handling is required in modelling.
- Topography sample is a derived surface model, not survey-grade field measurement.

## Licensing and Attribution

Most transport and boundary inputs originate from Hong Kong open data sources; OSM-derived assets follow OpenStreetMap/ODbL terms. Always keep source attribution in reports and derived outputs.

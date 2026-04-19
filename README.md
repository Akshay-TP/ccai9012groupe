# AI Detection of Urban Transport Inequality (Hong Kong)

This repository provides a reproducible workflow for measuring bus-access inequality in Hong Kong and generating candidate interventions.

The current version is a full refactor from straight-line proximity to **road-network shortest-path modelling** and **GNN-based stop optimization**.

## Quick Start

For a first-time full run, execute from repository root:

```bash
pip install -r requirements.txt
python src/01_fetch_data.py
python src/02_process_data.py
python src/03_compute_accessibility.py --remote-policy exclude_remote --distance-cap-m 2000
python src/04_ai_clustering.py
python src/05_visualise_results.py
python simulation/01_simulate_new_stops.py --remote-policy exclude_remote --distance-cap-m 2000
```

Road network note: `src/01_fetch_data.py` builds road graph files on first run, then reuses cache on normal reruns. Use `python src/01_fetch_data.py --refresh` only if you want updated OSM roads or if road graph files are missing/corrupted.

Main outputs are written to `output/` and `simulation/output/`.

## What Changed in This Refactor

Previous approach:
- Nearest-stop distances were calculated by Haversine distance.
- Simulation used heuristic ranking and spacing rules only.

Current approach:
- Accessibility distances are now computed along the Hong Kong road graph.
- Road graph is downloaded and cached in `data/raw/hk_roads_drive.graphml`.
- District and micro-grid scoring uses road-network distance to nearest stop.
- Simulation uses a graph neural network (GCN-style) with explicit reward and penalty terms.
- Interactive outputs now include road overlays where relevant.

## Policy Questions Addressed

1. Which districts face the highest road-based walking burden to bus access?
2. How unequal is access across districts after terrain and barrier-free adjustments?
3. Where should new stops be prioritized to maximize equity gains while avoiding invalid placements?

## Repository Structure

- `src/`
  - `01_fetch_data.py`: downloads all raw datasets and builds road graph
  - `02_process_data.py`: harmonizes operators and district transport aggregates
  - `03_compute_accessibility.py`: road-based micro-grid accessibility scoring
  - `04_ai_clustering.py`: district clustering from normalized features
  - `05_visualise_results.py`: maps, charts, 3D visuals, and summary report
  - `road_network.py`: shared road-graph helper utilities
- `data/`
  - `population_by_district.csv`: district population and area controls
  - `README.md`: dataset catalogue and schema notes
  - `raw/`: downloaded/generated source assets
- `output/`
  - accessibility, clustering, and visual outputs
- `simulation/`
  - `01_simulate_new_stops.py`: GNN stop optimization
  - `README.md`: simulation objective and interpretation
  - `output/`: simulation outputs

## Method Overview

### 1) Data Ingestion

`src/01_fetch_data.py` downloads:
- KMB, Citybus, NLB stops/routes/route-stop mappings
- District boundaries
- Hong Kong road network (OSM) as graph + edge GeoJSON
- Topography sample points from HK DTM
- Ramp proxy points from OSM/Overpass
- WorldPop metadata (and best-effort raster download)

### 2) District Aggregation

`src/02_process_data.py` creates district service aggregates:
- total stops
- route supply
- operator diversity
- merged with district population/area controls

### 3) Road-Based Accessibility Scoring

`src/03_compute_accessibility.py`:
- builds micro-grid cells within district polygons
- snaps cells/stops/ramps to road nodes
- computes shortest path distance from each cell to nearest stop (road only)
- applies terrain and ramp factors for effective walking burden
- population-weights cell outcomes back to district metrics
- computes composite accessibility score and Gini coefficient

### 4) Clustering

`src/04_ai_clustering.py` keeps the same K-Means analysis, now fed by road-based accessibility features.

### 5) Visualization

`src/05_visualise_results.py` produces:
- district choropleth (with optional road overlay)
- score bar chart
- cluster radar chart
- district/topography/micro-grid 3D HTML views
- consolidated 3D dashboard
- summary report CSV

### 6) GNN Simulation

`simulation/01_simulate_new_stops.py`:
- forms candidate graph from underserved micro-grid cells
- trains GCN-style model on reward/penalty objective
- penalizes water-cell placements
- penalizes remote-cell placements
- penalizes low-impact placements (proxy for equity score deterioration risk)
- enforces road-network spacing between selected candidates
- outputs candidate table, summary, map, and diagnostic images

## Accessibility Feature Set

Composite score uses normalized components:
- stop density (`stops_per_km2`)
- route density (`routes_per_km2`)
- per-capita service (`stops_per_10k`)
- inverse effective walking burden (`norm_walk_inv`)
- operator diversity (`norm_operator_div`)
- ramp coverage (`norm_ramp_cov`)
- terrain suitability (`norm_terrain_inv`)

Road distance is now the base proximity term for both district and micro-grid outcomes.

## Remote Areas: Distance Cap vs Exclusion

The pipeline supports both strategies in scoring and simulation:
- `exclude_remote`: drops unreachable/far-flung cells beyond threshold
- `distance_cap`: keeps all cells but caps distance at threshold

Recommendation for Hong Kong equity reporting:
- Use `exclude_remote` as default for core policy metrics.
- Reason: maritime/outlying extreme cells can dominate summary statistics without representing realistic stop intervention opportunities.

When to use `distance_cap`:
- If you need full territorial inclusion and want bounded influence rather than full removal.

CLI controls are available in both:
- `src/03_compute_accessibility.py`
- `simulation/01_simulate_new_stops.py`

## Installation

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

Notable packages:
- `geopandas`, `shapely` for spatial processing
- `networkx`, `osmnx` for road graph modelling
- `torch` for GNN training
- `folium`, `matplotlib`, `plotly` for visualization

## Running Order

Use the same command sequence shown in **Quick Start**. Conceptually, run in this exact order:

1. Fetch and cache all raw inputs (including the road graph).
2. Build district-level processed transport dataset.
3. Compute micro-grid and district accessibility metrics.
4. Cluster districts using the computed features.
5. Generate maps, charts, and 3D dashboard outputs.
6. Run stop-optimization simulation on underserved cells.

Road network run timing:
- First-time setup: run step 1 (it builds `data/raw/hk_roads_drive.graphml` and `data/raw/hk_roads_edges.geojson`).
- Normal reruns: you can skip rebuilding roads if those files already exist; cache mode will reuse them.
- Refresh case: rerun step 1 with `python src/01_fetch_data.py --refresh` when you want updated OSM roads or if graph files are missing/corrupted.

If you only change visualization styling, rerun step 5 only. If you change accessibility logic or parameters, rerun from step 3 onward.

## Key Outputs

Accessibility outputs:
- `output/accessibility_scores.csv`
- `output/micro_accessibility_grid.csv`
- `output/gini_coefficient.txt`
- `output/clustered_districts.csv`

Visualization outputs:
- `output/accessibility_map.html`
- `output/district_scores_bar.png`
- `output/cluster_profiles_radar.png`
- `output/district_accessibility_3d.html`
- `output/topography_3d.html`
- `output/micro_accessibility_3d.html`
- `output/3d_dashboard.html`
- `output/summary_report.csv`

Simulation outputs:
- `simulation/output/candidate_new_bus_stops.csv`
- `simulation/output/candidate_priority_summary.csv`
- `simulation/output/new_stop_candidates_map.html`
- `simulation/output/gnn_training_loss.png`
- `simulation/output/gnn_candidate_scores.png`
- `simulation/output/gnn_candidate_counts_by_district.png`

## Reproducibility Notes

- API responses evolve over time; archive `data/raw/` snapshots for report-grade reproducibility.
- Keep script order unchanged unless you intentionally rebuild all downstream outputs.
- For policy comparison experiments, hold `--remote-policy` and `--distance-cap-m` constant.

## Limitations

- Road graph uses OSM drivable network as proxy for walk-access connectivity.
- Ramp proxy coverage depends on OSM completeness.
- Micro-population allocation remains model-based within district totals.
- Candidate outputs are decision support and require engineering/field validation.

## License

MIT License (see `LICENSE`).

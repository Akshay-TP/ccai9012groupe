# AI Detection of Urban Transport Inequality (Hong Kong)

This repository provides a full analytical workflow for measuring and visualizing transport accessibility inequality across Hong Kong, with a strong focus on explainability and planning relevance.

The pipeline integrates open transport APIs (KMB, Citybus, NLB), district-level boundaries, topography samples, and barrier-free ramp proxies. Outputs include district scoring, inequality diagnostics, unsupervised clustering, interactive maps, and a simulation module for recommending candidate new bus stop locations.

## Project Scope

This project answers three policy-facing questions:

1. Which districts are currently better or worse served by bus transport?
2. How much inequality exists in accessibility across Hong Kong?
3. Where should planners prioritize new stop interventions for maximum impact?

To support these questions, the model combines supply-side service intensity with terrain and barrier-free accessibility effects at micro-grid scale.

## Key Enhancements in This Version

- Multi-operator bus integration: KMB + Citybus + New Lantao Bus.
- Tram removal: all tram-based features were removed to focus on a unified bus-access framework.
- Terrain-aware accessibility: elevation samples drive ruggedness penalties.
- Barrier-free factor: ramp-proxy proximity reduces effective walking burden.
- Fine spatial analysis: micro-grid modeling (finer than 200 m equivalent cell scale).
- 3D visual analytics: district-level, topography, and micro-grid interactive 3D outputs.
- Planning simulation: candidate new stop points with high/medium/low priority labels.

## End-to-End Workflow

The architecture is sequential and reproducible:

1. Data ingestion (`src/01_fetch_data.py`)
2. Spatial processing and district aggregation (`src/02_process_data.py`)
3. Accessibility scoring + inequality (`src/03_compute_accessibility.py`)
4. AI clustering (`src/04_ai_clustering.py`)
5. Visual and report generation (`src/05_visualise_results.py`)
6. New stop simulation (`simulation/01_simulate_new_stops.py`)

## Repository Structure

- `src/`
  - `01_fetch_data.py`: downloads and stores all raw inputs under `data/raw/`
  - `02_process_data.py`: harmonizes operators and aggregates district-level transport indicators
  - `03_compute_accessibility.py`: computes micro-grid and district-level accessibility metrics
  - `04_ai_clustering.py`: clusters districts by normalized feature profiles
  - `05_visualise_results.py`: generates 2D + 3D visuals and final summary report
- `data/`
  - `population_by_district.csv`: district population and area controls
  - `README.md`: detailed data catalogue and schema notes
  - `raw/`: downloaded and generated raw assets
- `output/`
  - all analysis outputs (scores, clusters, visuals, reports)
- `simulation/`
  - `01_simulate_new_stops.py`: stop candidate simulation logic
  - `README.md`: simulation method and interpretation
  - `output/`: simulation outputs

## Environment Setup

- Recommended Python: 3.10+
- Install dependencies:

```bash
pip install -r requirements.txt
```

Core libraries used:

- Data: pandas, numpy
- Spatial: geopandas, shapely
- ML: scikit-learn
- Visual: folium, matplotlib, plotly
- APIs: requests

## Execution Guide

Run from the project root in this exact sequence.

### 1) Fetch raw datasets

```bash
python src/01_fetch_data.py
```

Writes to `data/raw/`:

- KMB stops/routes/route-stops
- Citybus routes/route-stops/stops
- NLB routes/route-stops/stops
- District boundary GeoJSON
- Elevation sample table
- Ramp proxy JSON and point table
- Fine population metadata (and raster download attempt, when available)

### 2) Process and aggregate

```bash
python src/02_process_data.py
```

Creates:

- `output/district_transport_data.csv`

Main columns include:

- `total_stops`, `total_routes`, `population`, `area_km2`
- `kmb_stops`, `citybus_stops`, `nlb_stops`
- `operator_diversity`

### 3) Compute accessibility and inequality

```bash
python src/03_compute_accessibility.py
```

Creates:

- `output/accessibility_scores.csv`
- `output/micro_accessibility_grid.csv`
- `output/gini_coefficient.txt`

Highlights:

- Micro-grid effective walking distance per cell.
- Terrain and ramp adjustments to walking burden.
- Population-weighted district metrics.
- Composite accessibility score + rank.

### 4) Run AI clustering

```bash
python src/04_ai_clustering.py
```

Creates:

- `output/clustered_districts.csv`
- `output/elbow_plot.png`
- `output/silhouette_plot.png`

### 5) Generate maps, charts, and 3D outputs

```bash
python src/05_visualise_results.py
```

Creates:

- `output/accessibility_map.html`
- `output/district_scores_bar.png`
- `output/cluster_profiles_radar.png`
- `output/district_accessibility_3d.html`
- `output/topography_3d.html`
- `output/micro_accessibility_3d.html`
- `output/summary_report.csv`

### 6) Run stop-placement simulation

```bash
python simulation/01_simulate_new_stops.py
```

Creates:

- `simulation/output/candidate_new_bus_stops.csv`
- `simulation/output/candidate_priority_summary.csv`
- `simulation/output/new_stop_candidates_map.html`

## Accessibility Model Details

The composite score combines normalized features from district-level and micro-grid calculations.

Primary feature families:

- Service intensity: stop density and route density.
- Per-capita service: stops per 10k residents.
- Walkability outcome: inverse of effective walking distance.
- Operator resilience: service diversity across bus operators.
- Inclusive access: ramp coverage over modeled population.
- Terrain suitability: inverse ruggedness influence.

The inequality summary uses a Gini coefficient over district accessibility scores.

## 3D Visualization Layer

The project now includes three interactive 3D views:

1. District accessibility landscape
   - Axes: stops per km², routes per km², composite score.
   - Marker size: district population.
   - Purpose: compare service intensity and outcomes in one view.

2. Topography elevation sample
   - Axes: longitude, latitude, elevation.
   - Purpose: inspect terrain patterns underlying accessibility penalties.

3. Micro-grid burden surface (point cloud)
   - Axes: longitude, latitude, effective walking distance.
   - Marker size: micro-cell population.
   - Purpose: identify localized burden hotspots hidden by district averages.

## Interpretation Notes

- A district can have high stop density but still underperform if effective walking burden remains high.
- High ramp coverage tends to improve inclusion but does not fully offset steep terrain.
- Micro-grid results are critical for intervention targeting; district averages can hide within-district inequality.

## Assumptions and Limitations

- Straight-line proximity is used; full pedestrian network routing is not modeled.
- Ramp datasets are proxy-based and may be incomplete.
- Fine population assignment remains a model-based approximation.
- 3D visuals are exploratory analytics, not engineering-grade simulation geometry.
- Candidate stop outputs are decision support and require field validation.

## Reproducibility Notes

- API payloads can evolve over time; reruns may produce slightly different raw inputs.
- For consistent comparisons, archive generated `data/raw/` snapshots per run date.
- Keep script order unchanged to preserve schema compatibility.

## Data Sources

Detailed source links and file-level notes are documented in `data/README.md`.

## License

Released under the MIT License. See `LICENSE`.

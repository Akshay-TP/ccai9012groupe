# AI Detection of Urban Transport Inequality (Hong Kong)

This repository contains a small end-to-end pipeline (data collection → processing → accessibility scoring → clustering → visualisation) that analyses public transport accessibility across Hong Kong’s 18 districts.

The goal of the project is to quantify potential *transport inequality* using a set of interpretable accessibility metrics, then group districts into broad categories (high / medium / low accessibility) using unsupervised learning.

## What this project does

The pipeline produces district-level accessibility indicators from public open-data sources:

- Stop density and route density (per km²)
- Per-capita access (stops per 10,000 people)
- Estimated average walking distance to the nearest stop (grid-based approximation)
- A composite accessibility score (weighted combination of normalised metrics)
- A Gini coefficient to summarise inequality across districts
- K-Means clustering to categorise districts
- Maps and charts for reporting

## Repository structure

- **`src/`**
  - **`01_fetch_data.py`**: downloads raw datasets from Hong Kong open-data APIs into `data/raw/`
  - **`02_process_data.py`**: cleans/merges datasets and aggregates per district → `output/district_transport_data.csv`
  - **`03_compute_accessibility.py`**: computes accessibility metrics + composite score + Gini → `output/accessibility_scores.csv`
  - **`04_ai_clustering.py`**: runs K-Means clustering + evaluation plots → `output/clustered_districts.csv`
  - **`05_visualise_results.py`**: generates an interactive map and charts → `output/*.html`, `output/*.png`, `output/summary_report.csv`

- **`data/`**
  - **`population_by_district.csv`**: curated population + area dataset (used for per-capita and density metrics)
  - **`README.md`**: data catalogue and source links
  - **`raw/`** (generated, gitignored): raw downloads produced by `01_fetch_data.py`

- **`output/`** (generated, gitignored): all pipeline outputs

## Requirements

- Python 3.10+ recommended
- Dependencies listed in `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to run

Run the scripts in order from the project root.

### 1) Download raw datasets

```bash
python src/01_fetch_data.py
```

This creates:

- `data/raw/kmb_bus_stops.json`
- `data/raw/kmb_bus_routes.json`
- `data/raw/kmb_route_stops.json`
- `data/raw/tram_stops.csv`
- `data/raw/district_boundaries.json`

### 2) Process and aggregate data

```bash
python src/02_process_data.py
```

This creates:

- `output/district_transport_data.csv`

### 3) Compute accessibility metrics and inequality

```bash
python src/03_compute_accessibility.py
```

This creates:

- `output/accessibility_scores.csv`
- `output/gini_coefficient.txt`

Notes:

- The grid-based walking distance calculation can take a short while (it overlays a ~200m grid across each district).

### 4) Run clustering (“AI” step)

```bash
python src/04_ai_clustering.py
```

This creates:

- `output/clustered_districts.csv`
- `output/elbow_plot.png`
- `output/silhouette_plot.png`

### 5) Generate visualisations

```bash
python src/05_visualise_results.py
```

This creates:

- `output/accessibility_map.html` (interactive choropleth)
- `output/district_scores_bar.png`
- `output/cluster_profiles_radar.png`
- `output/summary_report.csv`

## Outputs (summary)

After running the full pipeline, the most important outputs are:

- **`output/summary_report.csv`**: consolidated table for analysis/reporting
- **`output/accessibility_map.html`**: interactive district map with pop-ups
- **`output/district_scores_bar.png`**: accessibility score comparison
- **`output/cluster_profiles_radar.png`**: feature profile per cluster
- **`output/gini_coefficient.txt`**: inequality summary statistic

## Data sources

All data sources are public and are documented in `data/README.md`. In brief:

- KMB bus stop / route datasets: `https://data.etabus.gov.hk`
- Tram stop dataset: `https://data.gov.hk`
- District boundary polygons: Home Affairs Department
- Population and area: curated from Census & Statistics Department tables (2021)

## Notes and assumptions

- The walking distance metric uses straight-line (Haversine) distance to the nearest stop, not road-network distance. It is intended as a practical approximation.
- The grid size is set to ~200m (implemented as `cell_size=0.002` degrees). This is a simplification and can be adjusted in `03_compute_accessibility.py`.
- K-Means is used because the dataset is small (18 districts) and interpretability is a priority.

## License

This project is released under the MIT License (see `LICENSE`).

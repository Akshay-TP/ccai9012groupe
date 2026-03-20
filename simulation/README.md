# Simulation Module

This module identifies potential new bus stop locations using micro-grid accessibility outputs and classifies recommendations into actionable priority tiers.

The goal is to convert inequality diagnostics into intervention candidates that planners can inspect, compare, and field-validate.

## Purpose

District averages alone are not enough for deployment decisions. This simulation works at micro-grid scale to find local hotspots where:

- effective walking burden is high, and
- affected population burden is high.

## Inputs

Required inputs:

- `output/micro_accessibility_grid.csv`
- `data/raw/kmb_bus_stops.json`
- `data/raw/citybus_stops.json`
- `data/raw/nlb_stops.json`
- `data/raw/district_boundaries.json`

Recommended upstream run order:

1. `python src/01_fetch_data.py`
2. `python src/02_process_data.py`
3. `python src/03_compute_accessibility.py`

## Script

- `simulation/01_simulate_new_stops.py`

Run command:

```bash
python simulation/01_simulate_new_stops.py
```

## Method Summary

The simulation follows five stages:

1. Candidate generation
	- Starts from micro-grid cells in each district.
2. Burden scoring
	- Computes a candidate score from effective walk excess and local cell population.
3. Existing-stop distance check
	- Avoids recommending points that are too close to current stops.
4. Greedy spacing filter
	- Keeps spatially distributed candidates and reduces redundancy.
5. Priority assignment
	- Labels retained candidates into high/medium/low tiers based on score quantiles.

## Burden Score Intuition

Each micro-cell gets a burden score that increases when:

- `effective_walk_dist_m` is above baseline threshold, and
- `cell_population` is larger.

Conceptually:

$$
	ext{burden} \propto \max(0, d_{\text{eff}} - d_{\text{baseline}}) \times \text{cell population}
$$

This design prioritizes interventions where accessibility deficits affect more residents.

## Outputs

Generated in `simulation/output/`:

- `candidate_new_bus_stops.csv`
  - candidate coordinates, district, priority tier, and scoring fields
- `candidate_priority_summary.csv`
  - priority counts and aggregate burden statistics
- `new_stop_candidates_map.html`
  - interactive map of current and proposed stop points

## Priority Labels

Default labels:

- High Priority
- Medium Priority
- Low Priority

Interpretation:

- High: strongest candidate impact under current model assumptions.
- Medium: meaningful but secondary interventions.
- Low: useful reserve set for phased or budget-constrained planning.

## Practical Use Notes

- Treat recommendations as pre-screening outputs.
- Perform field audits for geometry, road safety, and pedestrian path feasibility.
- Check legal/engineering constraints before deployment decisions.
- Use outputs in combination with district-level cluster findings and 3D burden/topography views.

## Tuning Suggestions

You can tune simulation behavior by adjusting:

- baseline effective walking threshold,
- minimum spacing distance,
- minimum distance from existing stops,
- priority quantile cutoffs.

Re-run the script after parameter changes and compare summary outputs to evaluate policy sensitivity.

## Limitations

- Uses straight-line distance proxies, not full walk network routing.
- Depends on raw stop coordinate quality from source APIs.
- Population is model-distributed at micro scale.
- Does not model road gradient safety, land ownership, curb geometry, or traffic operations.

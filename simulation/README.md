# Simulation Module (GNN Stop Optimization)

This module transforms micro-grid accessibility outputs into optimized bus-stop intervention candidates using a graph neural network.

## Why This Module Exists

District-level inequality metrics identify *where* problems exist, but deployment planning needs localized candidate points. This simulation layer converts cell-level burden into ranked intervention targets.

## Inputs

Required files:
- `output/micro_accessibility_grid.csv`
- `data/raw/hk_roads_drive.graphml`
- `data/raw/hk_topography_points.csv`
- `data/raw/kmb_bus_stops.json`
- `data/raw/citybus_stops.json`
- `data/raw/nlb_stops.json`

Recommended upstream sequence:
1. `python src/01_fetch_data.py`
2. `python src/02_process_data.py`
3. `python src/03_compute_accessibility.py`

## Script

- `simulation/01_simulate_new_stops.py`

Run command (recommended):

```bash
python simulation/01_simulate_new_stops.py --remote-policy exclude_remote --distance-cap-m 2000
```

## Optimization Pipeline

### 1) Candidate Pool Construction

From micro-grid cells, we compute:
- burden above target walk threshold: `max(effective_walk_dist_m - walk_target_m, 0)`
- population term: `sqrt(cell_population)`
- base impact: `burden * population_term`

Only positive-impact cells are retained, then limited to top pool size (`--max-candidates`).

### 2) Water and Remote Risk Tagging

- Water risk is inferred from lack of nearby topography support points.
- Remote risk comes from upstream `is_remote_cell` (or distance threshold fallback).

These are not blindly dropped by default in cap mode; they are heavily penalized in the objective.

### 3) Candidate Graph Construction

Candidates are treated as nodes in a local graph:
- edges connect nearest spatial neighbors
- edge weights decay by distance
- normalized adjacency is fed into GCN layers

### 4) Reward and Penalty Design

Reward signal favors placements with:
- high normalized burden
- high normalized population impact
- better ramp access context

Penalties include:
- `water_penalty`: discourages placing stops in likely water cells
- `remote_penalty`: discourages far-flung placements
- `score_drop_penalty`: discourages low-impact placement in already well-served cells (proxy for lowering equity gains)
- `low_impact_penalty`: regularizer against weak interventions

### 5) GNN Training Objective

Model predicts node placement probability. Loss combines:
- BCE fit to target desirability
- weighted placement penalties
- reward encouragement term

This yields a policy-oriented score that balances gain and feasibility.

### 6) Road-Network Spacing Constraint

Final ranked candidates are filtered with minimum spacing along road-network shortest paths (`--min-spacing-m`), reducing redundant nearby picks.

## Outputs

Generated in `simulation/output/`:
- `candidate_new_bus_stops.csv`
- `candidate_priority_summary.csv`
- `new_stop_candidates_map.html`
- `gnn_training_loss.png`
- `gnn_candidate_scores.png`
- `gnn_candidate_counts_by_district.png`

## Output Fields (candidate_new_bus_stops.csv)

Key columns:
- `candidate_id`
- `district`
- `lat`, `lon`
- `priority_level`
- `priority_score`
- `gnn_probability`
- `reward_signal`
- `score_drop_penalty`
- `water_penalty`
- `remote_penalty`
- `is_water_cell`
- `is_remote_cell`
- `effective_walk_dist_m`
- `nearest_stop_dist_m`
- `cell_population`

## Priority Tier Meaning

- High Priority: strongest utility under reward/penalty objective
- Medium Priority: moderate intervention value
- Low Priority: reserve options for phased deployment

## Tuning Parameters

Common controls:

| Parameter | Default | Meaning |
|---|---:|---|
| `--remote-policy` | `exclude_remote` | remote handling strategy |
| `--distance-cap-m` | `2000` | threshold/cap distance (m) |
| `--walk-target-m` | `350` | baseline acceptable walk distance |
| `--max-candidates` | `1600` | candidate pool size for GNN |
| `--min-spacing-m` | `350` | minimum road spacing between selected points |
| `--max-output` | `250` | final number of selected candidates |
| `--gnn-epochs` | `260` | training epochs |
| `--seed` | `42` | reproducibility seed |

## Remote-Island Handling Guidance

Recommended default for intervention planning:
- `--remote-policy exclude_remote`

Reason:
- It prevents extreme outlying cells from dominating the candidate budget.
- It better reflects practical deployment where road-served population impact is the objective.

Alternative:
- `--remote-policy distance_cap` keeps full geography but bounds influence.

## Practical Use Notes

- Treat outputs as pre-screening support, not automatic engineering decisions.
- Validate high-priority candidates with on-site audits and routing constraints.
- Use district-level accessibility outputs jointly with simulation outputs for balanced planning.

## Limitations

- Uses drivable road graph as access proxy.
- Water tagging is heuristic (topography support based), not cadastral shoreline truth.
- The score-drop penalty is a planning proxy, not a full causal simulation of district score trajectories.
- Does not model operational constraints (turning, curbside regulations, schedule impacts).

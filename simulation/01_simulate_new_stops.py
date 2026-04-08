"""
01_simulate_new_stops.py — GNN-based optimisation of new bus-stop locations
===========================================================================

This module proposes candidate bus-stop locations from micro-grid
accessibility outputs using a graph neural network (GNN).

Compared with the old heuristic ranker, this version:
1. Learns from a candidate-cell graph (GCN-style message passing).
2. Uses road-aware accessibility burden features.
3. Rewards high-impact placements.
4. Penalises low-impact placements that can weaken equity outcomes.
5. Penalises water-cell and remote-cell placements.

Inputs
------
1. output/micro_accessibility_grid.csv
2. data/raw/hk_roads_drive.graphml
3. data/raw/hk_topography_points.csv
4. Existing stop datasets in data/raw/

Outputs
-------
1. simulation/output/candidate_new_bus_stops.csv
2. simulation/output/candidate_priority_summary.csv
3. simulation/output/new_stop_candidates_map.html
4. simulation/output/gnn_training_loss.png
5. simulation/output/gnn_candidate_scores.png
6. simulation/output/gnn_candidate_counts_by_district.png

Author:  CCAI-9012 Group E
Date:    April 2026
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

import folium
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
SIM_OUTPUT_DIR = os.path.join(BASE_DIR, "simulation", "output")
ROADS_EDGES_PATH = os.path.join(RAW_DIR, "hk_roads_edges.geojson")

SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from road_network import (  # noqa: E402
    build_node_balltree,
    graph_node_arrays,
    load_road_graph,
    snap_points_to_nodes,
    to_simple_undirected,
)


@dataclass
class TrainingArtifacts:
    """Container for GNN training results."""
    predictions: np.ndarray
    loss_history: list[float]


def min_max(series: pd.Series) -> pd.Series:
    """Min-max scaling with numerical safety."""
    lo, hi = float(series.min()), float(series.max())
    return (series - lo) / (hi - lo + 1e-9)


def load_existing_stops() -> pd.DataFrame:
    """Load and combine operator stop tables for map context."""
    with open(os.path.join(RAW_DIR, "kmb_bus_stops.json"), "r", encoding="utf-8") as f:
        kmb = json.load(f)
    with open(os.path.join(RAW_DIR, "citybus_stops.json"), "r", encoding="utf-8") as f:
        city = json.load(f)
    with open(os.path.join(RAW_DIR, "nlb_stops.json"), "r", encoding="utf-8") as f:
        nlb = json.load(f)

    kmb_df = pd.DataFrame(kmb.get("data", []))
    if not kmb_df.empty:
        kmb_df = kmb_df.rename(columns={"stop": "stop_id", "lat": "lat", "long": "lon"})
        kmb_df["operator"] = "KMB"
        kmb_df = kmb_df[["stop_id", "lat", "lon", "operator"]]

    city_df = pd.DataFrame(city.get("data", []))
    if not city_df.empty:
        city_df = city_df.rename(columns={"stop": "stop_id", "lat": "lat", "long": "lon"})
        city_df["operator"] = "Citybus"
        city_df = city_df[["stop_id", "lat", "lon", "operator"]]

    nlb_df = pd.DataFrame(nlb.get("data", []))
    if not nlb_df.empty:
        nlb_df = nlb_df.rename(columns={"stopId": "stop_id", "latitude": "lat", "longitude": "lon"})
        nlb_df["operator"] = "NLB"
        nlb_df = nlb_df[["stop_id", "lat", "lon", "operator"]]

    all_stops = pd.concat([kmb_df, city_df, nlb_df], ignore_index=True)
    all_stops["lat"] = pd.to_numeric(all_stops["lat"], errors="coerce")
    all_stops["lon"] = pd.to_numeric(all_stops["lon"], errors="coerce")
    all_stops = all_stops.dropna(subset=["lat", "lon"])
    return all_stops.drop_duplicates(subset=["operator", "stop_id", "lat", "lon"])


def classify_water_cells(micro: pd.DataFrame, max_topo_dist_m: float = 500.0) -> pd.DataFrame:
    """
    Tag cells likely over water using topography-sample proximity.

    Cells without nearby topo support points are treated as water cells and
    are penalised during optimisation.
    """
    topo_path = os.path.join(RAW_DIR, "hk_topography_points.csv")
    if not os.path.exists(topo_path):
        micro["is_water_cell"] = 0
        return micro

    topo = pd.read_csv(topo_path)
    topo["lat"] = pd.to_numeric(topo.get("lat"), errors="coerce")
    topo["lon"] = pd.to_numeric(topo.get("lon"), errors="coerce")
    topo = topo.dropna(subset=["lat", "lon"])

    if topo.empty:
        micro["is_water_cell"] = 0
        return micro

    # 0.002 degrees is about 200-220m in Hong Kong latitude.
    bin_size = 0.002
    topo_bins: set[tuple[int, int]] = set()
    for lat, lon in zip(topo["lat"].values, topo["lon"].values):
        topo_bins.add((int(np.floor(lat / bin_size)), int(np.floor(lon / bin_size))))

    lat_bins = np.floor(micro["lat"].values / bin_size).astype(int)
    lon_bins = np.floor(micro["lon"].values / bin_size).astype(int)

    supported = np.zeros(len(micro), dtype=bool)
    for dlat in range(-1, 2):
        for dlon in range(-1, 2):
            keys = zip(lat_bins + dlat, lon_bins + dlon)
            hits = np.array([k in topo_bins for k in keys])
            supported |= hits

    micro["is_water_cell"] = (~supported).astype(int)
    water_n = int(micro["is_water_cell"].sum())
    print(
        f"    Water tagging: {water_n:,} cells tagged as water/no-topography "
        f"(~{max_topo_dist_m:.0f} m proximity check)"
    )
    return micro


def build_candidate_pool(
    micro: pd.DataFrame,
    max_candidates: int,
    walk_target_m: float,
    remote_policy: str,
    distance_cap_m: float,
) -> pd.DataFrame:
    """Create candidate pool and compute base burden components."""
    work = micro.copy()

    work["nearest_stop_dist_m"] = pd.to_numeric(work["nearest_stop_dist_m"], errors="coerce")
    work["effective_walk_dist_m"] = pd.to_numeric(work["effective_walk_dist_m"], errors="coerce")
    work["cell_population"] = pd.to_numeric(work["cell_population"], errors="coerce")
    work["terrain_penalty"] = pd.to_numeric(work["terrain_penalty"], errors="coerce")
    work["nearest_ramp_dist_m"] = pd.to_numeric(work["nearest_ramp_dist_m"], errors="coerce")

    work = work.dropna(
        subset=["lat", "lon", "nearest_stop_dist_m", "effective_walk_dist_m", "cell_population"]
    )

    if "is_remote_cell" in work.columns:
        work["is_remote_cell"] = work["is_remote_cell"].astype(int)
    else:
        work["is_remote_cell"] = (work["nearest_stop_dist_m"] > distance_cap_m).astype(int)

    if remote_policy == "exclude_remote":
        before = len(work)
        work = work[work["is_remote_cell"] == 0].copy()
        print(f"    Remote filter: removed {before - len(work):,} remote cells (exclude policy)")
    elif remote_policy == "distance_cap":
        work["nearest_stop_dist_m"] = np.minimum(work["nearest_stop_dist_m"], distance_cap_m)

    work["burden_m"] = np.maximum(work["effective_walk_dist_m"] - walk_target_m, 0.0)
    work["pop_term"] = np.sqrt(np.maximum(work["cell_population"], 0.0))
    work["base_impact"] = work["burden_m"] * work["pop_term"]

    if (work["base_impact"] > 0).any():
        work = work[work["base_impact"] > 0].copy()

    if len(work) > max_candidates:
        work = work.nlargest(max_candidates, "base_impact").copy()

    return work.reset_index(drop=True)


def attach_road_nodes(candidates: pd.DataFrame) -> tuple[pd.DataFrame, nx.Graph]:
    """Snap candidate points to road nodes and return graph context."""
    graph_multi = load_road_graph(RAW_DIR)
    graph = to_simple_undirected(graph_multi)

    node_ids, node_lats, node_lons = graph_node_arrays(graph)
    tree = build_node_balltree(node_lats, node_lons)

    snapped, connector_m = snap_points_to_nodes(
        candidates["lat"].values,
        candidates["lon"].values,
        node_ids,
        tree,
    )
    candidates = candidates.copy()
    candidates["road_node"] = snapped
    candidates["road_connector_m"] = connector_m
    return candidates, graph


def build_adjacency(candidates: pd.DataFrame, k_neighbors: int = 8) -> np.ndarray:
    """Build normalized dense adjacency matrix for GCN training."""
    coords = np.radians(candidates[["lat", "lon"]].values)
    tree = BallTree(coords, metric="haversine")
    n = len(candidates)
    k = min(k_neighbors + 1, n)

    d_rad, idx = tree.query(coords, k=k)
    d_m = d_rad * 6_371_000.0

    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j_pos in range(1, k):
            j = idx[i, j_pos]
            dist = d_m[i, j_pos]
            weight = float(np.exp(-dist / 600.0))
            adj[i, j] = max(adj[i, j], weight)
            adj[j, i] = max(adj[j, i], weight)

    np.fill_diagonal(adj, 1.0)
    deg = np.sum(adj, axis=1)
    deg_inv_sqrt = np.diag(1.0 / np.sqrt(deg + 1e-9))
    a_hat = deg_inv_sqrt @ adj @ deg_inv_sqrt
    return a_hat.astype(np.float32)


def build_reward_table(candidates: pd.DataFrame) -> pd.DataFrame:
    """Create reward and penalty terms used by the optimization objective."""
    out = candidates.copy()

    out["burden_norm"] = min_max(out["burden_m"])
    out["population_norm"] = min_max(out["cell_population"])

    ramp_prox = 1.0 / (1.0 + np.maximum(out["nearest_ramp_dist_m"], 0.0) / 120.0)
    out["ramp_access_norm"] = min_max(pd.Series(ramp_prox, index=out.index))

    out["reward_signal"] = (
        0.55 * out["burden_norm"]
        + 0.35 * out["population_norm"]
        + 0.10 * out["ramp_access_norm"]
    )

    # Penalty for likely low-impact placements in already well-served areas.
    out["score_drop_penalty"] = (
        ((out["nearest_stop_dist_m"] < 250.0) & (out["effective_walk_dist_m"] < 300.0)).astype(float)
        * 0.90
    )

    out["low_impact_penalty"] = (1.0 - out["burden_norm"]) * 0.30
    out["water_penalty"] = out["is_water_cell"].astype(float) * 1.30
    out["remote_penalty"] = out["is_remote_cell"].astype(float) * 0.85

    raw_target = (
        out["reward_signal"]
        - out["score_drop_penalty"]
        - out["low_impact_penalty"]
        - out["water_penalty"]
        - out["remote_penalty"]
    )
    out["target_prob"] = 1.0 / (1.0 + np.exp(-3.0 * raw_target))
    return out


def train_gnn(candidates: pd.DataFrame, adjacency: np.ndarray, seed: int, epochs: int) -> TrainingArtifacts:
    """Train GCN and return node probabilities."""
    import importlib

    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
    F = importlib.import_module("torch.nn.functional")

    class SimpleGCN(nn.Module):
        """Lightweight two-layer GCN using dense normalized adjacency."""

        def __init__(self, in_dim: int, hidden_dim: int = 32) -> None:
            super().__init__()
            self.lin1 = nn.Linear(in_dim, hidden_dim)
            self.lin2 = nn.Linear(hidden_dim, 1)

        def forward(self, x, a_hat):
            h = torch.matmul(a_hat, x)
            h = F.relu(self.lin1(h))
            h = torch.matmul(a_hat, h)
            out = torch.sigmoid(self.lin2(h)).squeeze(-1)
            return out

    torch.manual_seed(seed)
    np.random.seed(seed)

    feature_cols = [
        "burden_norm",
        "population_norm",
        "ramp_access_norm",
        "terrain_penalty",
        "is_water_cell",
        "is_remote_cell",
        "nearest_stop_dist_m",
    ]

    # Keep feature magnitudes stable for training.
    f = candidates[feature_cols].copy()
    f["terrain_penalty"] = min_max(f["terrain_penalty"])
    f["nearest_stop_dist_m"] = min_max(f["nearest_stop_dist_m"])

    x = torch.tensor(f.values, dtype=torch.float32)
    a_hat = torch.tensor(adjacency, dtype=torch.float32)
    target = torch.tensor(candidates["target_prob"].values, dtype=torch.float32)

    reward = torch.tensor(candidates["reward_signal"].values, dtype=torch.float32)
    water = torch.tensor(candidates["is_water_cell"].values, dtype=torch.float32)
    remote = torch.tensor(candidates["is_remote_cell"].values, dtype=torch.float32)
    score_drop = torch.tensor(candidates["score_drop_penalty"].values, dtype=torch.float32)

    model = SimpleGCN(in_dim=x.shape[1], hidden_dim=32)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    loss_history: list[float] = []
    for _ in range(epochs):
        pred = model(x, a_hat)
        bce = F.binary_cross_entropy(pred, target)

        penalty_term = (
            1.00 * torch.mean(pred * water)
            + 0.70 * torch.mean(pred * remote)
            + 0.85 * torch.mean(pred * score_drop)
        )
        reward_term = torch.mean(pred * reward)

        loss = bce + penalty_term - 0.55 * reward_term

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_history.append(float(loss.detach().cpu().item()))

    with torch.no_grad():
        probs = model(x, a_hat).detach().cpu().numpy()

    return TrainingArtifacts(predictions=probs, loss_history=loss_history)


def road_spacing_filter(
    ranked: pd.DataFrame,
    graph: nx.Graph,
    min_spacing_m: float,
    max_out: int,
) -> pd.DataFrame:
    """Greedy selection with road-network spacing enforcement."""
    selected_rows = []
    selected_nodes: list[str] = []
    nearby_cache: dict[str, set[str]] = {}

    for _, row in ranked.iterrows():
        node = str(row["road_node"])

        if not selected_nodes:
            selected_rows.append(row)
            selected_nodes.append(node)
            if len(selected_rows) >= max_out:
                break
            continue

        is_too_close = False
        for sel_node in selected_nodes:
            if sel_node == node:
                is_too_close = True
                break

            if sel_node not in nearby_cache:
                lengths = nx.single_source_dijkstra_path_length(
                    graph,
                    sel_node,
                    cutoff=min_spacing_m,
                    weight="length",
                )
                nearby_cache[sel_node] = {str(k) for k in lengths.keys()}

            if node in nearby_cache[sel_node]:
                is_too_close = True
                break

        if is_too_close:
            continue

        selected_rows.append(row)
        selected_nodes.append(node)

        if len(selected_rows) >= max_out:
            break

    if not selected_rows:
        return ranked.head(0).copy()

    return pd.DataFrame(selected_rows)


def assign_priority(score_series: pd.Series) -> pd.Series:
    """Map continuous utility scores to high / medium / low tiers."""
    q_high = score_series.quantile(0.67)
    q_med = score_series.quantile(0.33)

    def _label(x: float) -> str:
        if x >= q_high:
            return "High Priority"
        if x >= q_med:
            return "Medium Priority"
        return "Low Priority"

    return score_series.apply(_label)


def create_candidate_map(candidates: pd.DataFrame, existing_stops: pd.DataFrame) -> folium.Map:
    """Create interactive map showing existing stops and optimized candidates."""
    m = folium.Map(location=[22.35, 114.15], zoom_start=11, tiles="CartoDB positron")

    if os.path.exists(ROADS_EDGES_PATH):
        roads = gpd.read_file(ROADS_EDGES_PATH)
        if not roads.empty:
            if len(roads) > 25000:
                roads = roads.sample(25000, random_state=42)
            folium.GeoJson(
                data=json.loads(roads.to_json()),
                name="Road Network",
                style_function=lambda _: {
                    "color": "#475569",
                    "weight": 1,
                    "opacity": 0.35,
                },
            ).add_to(m)

    for _, s in existing_stops.iterrows():
        folium.CircleMarker(
            location=[s["lat"], s["lon"]],
            radius=1,
            color="#60a5fa",
            fill=True,
            fill_opacity=0.35,
            weight=0,
        ).add_to(m)

    colors = {
        "High Priority": "#dc2626",
        "Medium Priority": "#f59e0b",
        "Low Priority": "#22c55e",
    }

    for _, c in candidates.iterrows():
        water_tag = "Yes" if int(c["is_water_cell"]) == 1 else "No"
        folium.CircleMarker(
            location=[c["lat"], c["lon"]],
            radius=5,
            color=colors.get(c["priority_level"], "#6b7280"),
            fill=True,
            fill_opacity=0.92,
            popup=(
                f"District: {c['district']}<br>"
                f"Priority: {c['priority_level']}<br>"
                f"GNN Probability: {c['gnn_probability']:.3f}<br>"
                f"Utility Score: {c['priority_score']:.3f}<br>"
                f"Effective Walk Dist: {c['effective_walk_dist_m']:.0f} m<br>"
                f"Road Dist to Stop: {c['nearest_stop_dist_m']:.0f} m<br>"
                f"Cell Population: {c['cell_population']:.1f}<br>"
                f"Water Cell: {water_tag}"
            ),
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    return m


def save_diagnostic_images(candidates: pd.DataFrame, loss_history: list[float]) -> None:
    """Save training diagnostics and candidate score visuals."""
    loss_path = os.path.join(SIM_OUTPUT_DIR, "gnn_training_loss.png")
    score_path = os.path.join(SIM_OUTPUT_DIR, "gnn_candidate_scores.png")
    district_path = os.path.join(SIM_OUTPUT_DIR, "gnn_candidate_counts_by_district.png")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(loss_history, color="#1d4ed8", linewidth=2)
    ax.set_title("GNN Training Loss", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Objective Value")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.8))
    colors = {"High Priority": "#dc2626", "Medium Priority": "#f59e0b", "Low Priority": "#22c55e"}
    for level, grp in candidates.groupby("priority_level"):
        ax.scatter(
            grp["gnn_probability"],
            grp["priority_score"],
            label=level,
            s=22,
            alpha=0.72,
            color=colors.get(level, "#6b7280"),
        )
    ax.set_title("GNN Output vs Final Utility", fontweight="bold")
    ax.set_xlabel("GNN Probability")
    ax.set_ylabel("Priority Utility Score")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(score_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    district_counts = (
        candidates.groupby(["district", "priority_level"]).size().reset_index(name="count")
    )
    pivot = district_counts.pivot(index="district", columns="priority_level", values="count").fillna(0)
    pivot = pivot.sort_values(by=list(pivot.columns), ascending=False)

    fig, ax = plt.subplots(figsize=(12, 8))
    pivot.plot(kind="bar", stacked=True, ax=ax, color=["#dc2626", "#f59e0b", "#22c55e"])
    ax.set_title("Optimized Candidate Counts by District", fontweight="bold")
    ax.set_xlabel("District")
    ax.set_ylabel("Candidate Count")
    ax.legend(title="Priority")
    fig.tight_layout()
    fig.savefig(district_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(
    remote_policy: str,
    distance_cap_m: float,
    walk_target_m: float,
    max_candidates: int,
    min_spacing_m: float,
    max_output: int,
    gnn_epochs: int,
    random_seed: int,
) -> None:
    print("=" * 60)
    print("Simulation  ·  GNN bus-stop optimisation")
    print("=" * 60)

    os.makedirs(SIM_OUTPUT_DIR, exist_ok=True)

    micro_path = os.path.join(OUTPUT_DIR, "micro_accessibility_grid.csv")
    if not os.path.exists(micro_path):
        raise FileNotFoundError(
            "Missing micro_accessibility_grid.csv. Run src/03_compute_accessibility.py first."
        )

    print("\n[1/6]  Loading micro-grid data and tagging water cells …")
    micro = pd.read_csv(micro_path)
    micro = classify_water_cells(micro)

    print("\n[2/6]  Building candidate pool …")
    candidates = build_candidate_pool(
        micro=micro,
        max_candidates=max_candidates,
        walk_target_m=walk_target_m,
        remote_policy=remote_policy,
        distance_cap_m=distance_cap_m,
    )
    if candidates.empty:
        print("No valid candidates after filtering. Check remote policy or upstream outputs.")
        return

    print("\n[3/6]  Snapping candidates to road graph and building candidate graph …")
    candidates, road_graph = attach_road_nodes(candidates)
    adjacency = build_adjacency(candidates, k_neighbors=8)

    print("\n[4/6]  Training GNN with reward/penalty objective …")
    candidates = build_reward_table(candidates)
    artifacts = train_gnn(
        candidates=candidates,
        adjacency=adjacency,
        seed=random_seed,
        epochs=gnn_epochs,
    )

    candidates["gnn_probability"] = artifacts.predictions
    candidates["priority_score"] = (
        candidates["gnn_probability"] * (candidates["reward_signal"] + 1e-6)
        - candidates["score_drop_penalty"]
        - candidates["low_impact_penalty"]
        - candidates["water_penalty"]
        - candidates["remote_penalty"]
    )

    ranked = candidates.sort_values("priority_score", ascending=False).copy()

    print("\n[5/6]  Applying road-network spacing and assigning priority tiers …")
    selected = road_spacing_filter(
        ranked=ranked,
        graph=road_graph,
        min_spacing_m=min_spacing_m,
        max_out=max_output,
    )

    if selected.empty:
        print("No candidate points were selected after spacing filter.")
        return

    selected["priority_level"] = assign_priority(selected["priority_score"])
    selected["candidate_id"] = [f"CAND_{i:04d}" for i in range(1, len(selected) + 1)]

    ordered_cols = [
        "candidate_id",
        "district",
        "lat",
        "lon",
        "priority_level",
        "priority_score",
        "gnn_probability",
        "reward_signal",
        "score_drop_penalty",
        "low_impact_penalty",
        "water_penalty",
        "remote_penalty",
        "is_water_cell",
        "is_remote_cell",
        "effective_walk_dist_m",
        "nearest_stop_dist_m",
        "cell_population",
        "terrain_penalty",
        "nearest_ramp_dist_m",
        "road_node",
        "road_connector_m",
    ]
    selected_out = selected[ordered_cols].copy()

    summary = (
        selected_out.groupby(["district", "priority_level"]).size().reset_index(name="candidate_count")
        .sort_values(["priority_level", "candidate_count"], ascending=[True, False])
    )

    print("\n[6/6]  Saving outputs (CSV, map, and images) …")
    existing = load_existing_stops()

    candidates_csv = os.path.join(SIM_OUTPUT_DIR, "candidate_new_bus_stops.csv")
    summary_csv = os.path.join(SIM_OUTPUT_DIR, "candidate_priority_summary.csv")
    map_html = os.path.join(SIM_OUTPUT_DIR, "new_stop_candidates_map.html")

    selected_out.to_csv(candidates_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    candidate_map = create_candidate_map(selected_out, existing)
    candidate_map.save(map_html)

    save_diagnostic_images(selected_out, artifacts.loss_history)

    print(f"\nCandidate locations saved: {candidates_csv}")
    print(f"Priority summary saved:    {summary_csv}")
    print(f"Interactive map saved:     {map_html}")
    print(f"Districts represented:     {selected_out['district'].nunique()}")
    print(f"Total candidates:          {len(selected_out)}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GNN-based bus-stop candidate optimisation on micro accessibility grid.",
    )
    parser.add_argument(
        "--remote-policy",
        choices=["exclude_remote", "distance_cap"],
        default="exclude_remote",
        help="How to handle remote/far-flung cells in candidate pool.",
    )
    parser.add_argument(
        "--distance-cap-m",
        type=float,
        default=2000.0,
        help="Threshold/cap for remote cells (m).",
    )
    parser.add_argument(
        "--walk-target-m",
        type=float,
        default=350.0,
        help="Target effective walking distance baseline (m).",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=1600,
        help="Maximum candidate cells considered by the GNN.",
    )
    parser.add_argument(
        "--min-spacing-m",
        type=float,
        default=350.0,
        help="Minimum spacing between selected candidates (road distance).",
    )
    parser.add_argument(
        "--max-output",
        type=int,
        default=250,
        help="Maximum number of final selected candidates.",
    )
    parser.add_argument(
        "--gnn-epochs",
        type=int,
        default=260,
        help="Training epochs for the GNN model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    main(
        remote_policy=args.remote_policy,
        distance_cap_m=args.distance_cap_m,
        walk_target_m=args.walk_target_m,
        max_candidates=args.max_candidates,
        min_spacing_m=args.min_spacing_m,
        max_output=args.max_output,
        gnn_epochs=args.gnn_epochs,
        random_seed=args.seed,
    )

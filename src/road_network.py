"""
road_network.py — Shared utilities for Hong Kong road-network modelling
=======================================================================

This module centralises graph loading, point-to-node snapping, and
shortest-path distance helpers used by both accessibility scoring and
simulation.

Author:  CCAI-9012 Group E
Date:    April 2026
"""

from __future__ import annotations

import os
from typing import Iterable

import networkx as nx
import numpy as np
from sklearn.neighbors import BallTree

EARTH_RADIUS_M = 6_371_000.0


def load_road_graph(raw_dir: str, filename: str = "hk_roads_drive.graphml") -> nx.MultiDiGraph:
    """
    Load the persisted Hong Kong road graph.

    The graph is created in src/01_fetch_data.py and saved under data/raw/.
    """
    path = os.path.join(raw_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Road graph not found at {path}. Run src/01_fetch_data.py first."
        )

    graph = nx.read_graphml(path)
    if graph.number_of_nodes() == 0:
        raise ValueError("Road graph file is empty (0 nodes).")

    # Ensure edge lengths are numeric for weighted shortest paths.
    for _, _, data in graph.edges(data=True):
        data["length"] = float(data.get("length", 0.0) or 0.0)

    return graph


def to_simple_undirected(graph: nx.MultiDiGraph) -> nx.Graph:
    """
    Convert a MultiDiGraph to a simple undirected weighted graph.

    We keep the shortest edge when multiple parallel edges exist between
    the same node pair. This is suitable for walking/catchment analysis.
    """
    simple = nx.Graph()

    for node_id, data in graph.nodes(data=True):
        try:
            x = float(data.get("x"))
            y = float(data.get("y"))
        except (TypeError, ValueError):
            continue
        simple.add_node(str(node_id), x=x, y=y)

    for u, v, data in graph.edges(data=True):
        u = str(u)
        v = str(v)
        if not (simple.has_node(u) and simple.has_node(v)):
            continue

        length = float(data.get("length", 0.0) or 0.0)
        if length <= 0:
            continue

        if simple.has_edge(u, v):
            if length < simple[u][v]["length"]:
                simple[u][v]["length"] = length
        else:
            simple.add_edge(u, v, length=length)

    if simple.number_of_nodes() == 0 or simple.number_of_edges() == 0:
        raise ValueError("Road graph conversion failed; no valid nodes/edges.")

    return simple


def graph_node_arrays(graph: nx.Graph) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return parallel arrays: node_ids, node_lats, node_lons."""
    node_ids: list[str] = []
    lats: list[float] = []
    lons: list[float] = []

    for node_id, data in graph.nodes(data=True):
        try:
            lon = float(data.get("x"))
            lat = float(data.get("y"))
        except (TypeError, ValueError):
            continue
        node_ids.append(str(node_id))
        lats.append(lat)
        lons.append(lon)

    if not node_ids:
        raise ValueError("No graph nodes with numeric coordinates were found.")

    return np.array(node_ids, dtype=object), np.array(lats), np.array(lons)


def build_node_balltree(node_lats: np.ndarray, node_lons: np.ndarray) -> BallTree:
    """Build a haversine BallTree over graph-node coordinates."""
    coords_rad = np.radians(np.c_[node_lats, node_lons])
    return BallTree(coords_rad, metric="haversine")


def snap_points_to_nodes(
    lats: np.ndarray,
    lons: np.ndarray,
    node_ids: np.ndarray,
    balltree: BallTree,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Snap geographic points to nearest graph nodes.

    Returns:
    1) nearest node IDs
    2) connector distances in metres (point -> nearest node)
    """
    query = np.radians(np.c_[lats, lons])
    dist_rad, idx = balltree.query(query, k=1)
    nearest_ids = node_ids[idx.ravel()]
    connector_m = dist_rad.ravel() * EARTH_RADIUS_M
    return nearest_ids, connector_m


def multi_source_shortest_path_lengths(
    graph: nx.Graph,
    source_nodes: Iterable[str],
) -> dict[str, float]:
    """Compute shortest-path length (metres) to nearest source node."""
    sources = [str(s) for s in source_nodes if graph.has_node(str(s))]
    if not sources:
        return {}
    return nx.multi_source_dijkstra_path_length(graph, sources, weight="length")


def lookup_distances(
    snapped_node_ids: np.ndarray,
    distance_map: dict[str, float],
    connector_m: np.ndarray | None = None,
    unreachable_value: float = np.inf,
) -> np.ndarray:
    """
    Convert snapped-node IDs into distance array.

    If connector distances are provided, they are added to the network
    distance to better represent off-network start/end offsets.
    """
    out = np.full(len(snapped_node_ids), unreachable_value, dtype=float)
    for i, node_id in enumerate(snapped_node_ids):
        value = distance_map.get(str(node_id))
        if value is None:
            continue
        out[i] = float(value)

    if connector_m is not None:
        out = out + connector_m

    return out

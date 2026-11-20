"""Structural and weight features. None of them solve the weighted shortest-path problem."""
from __future__ import annotations

import math
import statistics
import time

from algoselect.graphs import GraphInstance


STRUCTURAL = (
    "vertices",
    "edges",
    "density",
    "average_degree",
    "min_degree",
    "max_degree",
    "degree_variance",
    "isolated_nodes",
    "components",
    "largest_component",
    "clustering",
    "degree_p50",
    "degree_p90",
    "bfs_depth",
    "grid_score",
)
WEIGHTS = (
    "weight_min",
    "weight_max",
    "weight_mean",
    "weight_std",
    "weight_median",
    "weight_bound",
)
BASIC_COUNTS = ("vertices", "edges")
FULL = STRUCTURAL + WEIGHTS


def _components(adj: list[list[tuple[int, int]]]) -> tuple[int, int, int]:
    n = len(adj)
    seen = [False] * n
    count = 0
    largest = 0
    isolated = 0
    for start in range(n):
        if not adj[start]:
            isolated += 1
        if seen[start]:
            continue
        count += 1
        stack = [start]
        seen[start] = True
        size = 0
        while stack:
            u = stack.pop()
            size += 1
            for v, _w in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        largest = max(largest, size)
    return count, largest, isolated


def _clustering(adj: list[list[tuple[int, int]]]) -> float:
    neighbor_sets = [set(v for v, _w in nbrs) for nbrs in adj]
    coeffs: list[float] = []
    for u, nbrs in enumerate(neighbor_sets):
        k = len(nbrs)
        if k < 2:
            continue
        nodes = list(nbrs)
        links = 0
        for i, v in enumerate(nodes):
            for w in nodes[i + 1 :]:
                if w in neighbor_sets[v]:
                    links += 1
        coeffs.append((2.0 * links) / (k * (k - 1)))
    return sum(coeffs) / len(coeffs) if coeffs else 0.0


def _bfs_depth(adj: list[list[tuple[int, int]]]) -> float:
    if not adj:
        return 0.0
    dist = [-1] * len(adj)
    dist[0] = 0
    queue = [0]
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        for v, _w in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                queue.append(v)
    reached = [d for d in dist if d >= 0]
    return float(max(reached) if reached else 0)


def extract(graph: GraphInstance) -> dict[str, float]:
    n = graph.n
    undirected = graph.m_undirected
    degrees = [len(nbrs) for nbrs in graph.adj]
    components, largest, isolated = _components(graph.adj)
    weights = [w for _u, _v, w in graph.edges[::2]] or [0]
    max_possible = n * (n - 1) / 2
    density = undirected / max_possible if max_possible else 0.0
    avg = sum(degrees) / n if n else 0.0
    var = sum((d - avg) ** 2 for d in degrees) / n if n else 0.0
    ordered = sorted(degrees)
    def q(p: float) -> float:
        if not ordered:
            return 0.0
        idx = min(len(ordered) - 1, int(p * (len(ordered) - 1)))
        return float(ordered[idx])
    w_mean = sum(weights) / len(weights)
    w_var = sum((w - w_mean) ** 2 for w in weights) / len(weights)
    grid_score = 1.0 if max(degrees or [0]) <= 4 and math.sqrt(var) < 1.0 else 0.0
    return {
        "vertices": float(n),
        "edges": float(undirected),
        "density": density,
        "average_degree": avg,
        "min_degree": float(min(degrees) if degrees else 0),
        "max_degree": float(max(degrees) if degrees else 0),
        "degree_variance": var,
        "isolated_nodes": float(isolated),
        "components": float(components),
        "largest_component": float(largest),
        "clustering": _clustering(graph.adj),
        "degree_p50": q(0.5),
        "degree_p90": q(0.9),
        "bfs_depth": _bfs_depth(graph.adj),
        "grid_score": grid_score,
        "weight_min": float(min(weights)),
        "weight_max": float(max(weights)),
        "weight_mean": w_mean,
        "weight_std": math.sqrt(w_var),
        "weight_median": float(statistics.median(weights)),
        "weight_bound": float(max(weights)),
    }


def timed_extract(graph: GraphInstance) -> tuple[dict[str, float], float]:
    t0 = time.perf_counter()
    feats = extract(graph)
    return feats, (time.perf_counter() - t0) * 1000.0


def vector(feats: dict[str, float], names: tuple[str, ...]) -> list[float]:
    return [float(feats[name]) for name in names]

"""Feasibility, timing, and memory measurement. Timing and memory are separate calls."""
from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from math import inf, isclose

from algoselect.algorithms import bellman_ford, bfs_shortest_paths, dial, dijkstra_array, dijkstra_heap
from algoselect.graphs import GraphInstance


def feasible(graph: GraphInstance) -> list[str]:
    """Algorithms that solve the same task as the graph's weights allow."""
    weights = [w for _u, _v, w in graph.edges] or [1]
    nonnegative = min(weights) >= 0
    integer = all(float(w).is_integer() for w in weights)
    names: list[str] = []
    if graph.unweighted and all(w == 1 for w in weights):
        names.append("bfs")
    if nonnegative:
        names.append("dijkstra_heap")
        names.append("dijkstra_array")
        if integer:
            names.append("dial")
    names.append("bellman_ford")
    return names


def run_algorithm(name: str, graph: GraphInstance, source: int = 0) -> list[float]:
    if name == "bfs":
        return bfs_shortest_paths(graph.adj, source)
    if name == "dijkstra_heap":
        return dijkstra_heap(graph.adj, source)
    if name == "dijkstra_array":
        return dijkstra_array(graph.adj, source)
    if name == "bellman_ford":
        return bellman_ford(graph.n, graph.edges, source)
    if name == "dial":
        bound = max((w for _u, _v, w in graph.edges), default=0)
        return dial(graph.adj, source, bound)
    raise KeyError(name)


def results_match(actual: list[float], expected: list[float], tolerance: float = 1e-9) -> bool:
    if len(actual) != len(expected):
        return False
    for a, e in zip(actual, expected):
        if a == inf and e == inf:
            continue
        if a == inf or e == inf:
            return False
        if not isclose(a, e, rel_tol=tolerance, abs_tol=tolerance):
            return False
    return True


def measure_runtime_ms(fn, repeats: int = 3) -> list[float]:
    fn()
    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - t0) / 1_000_000)
    return samples


def measure_peak_kb(fn) -> float:
    tracemalloc.start()
    try:
        fn()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / 1024.0


@dataclass
class Observation:
    graph_id: str
    graph_family: str
    seed: int
    vertices: int
    edges: int
    algorithm: str
    run: int
    runtime_ms: float
    peak_memory_kb: float
    correct: bool


def benchmark_graph(graph: GraphInstance, *, repeats: int = 2, source: int = 0) -> list[Observation]:
    names = feasible(graph)
    reference_name = "dijkstra_heap" if "dijkstra_heap" in names else names[0]
    reference = run_algorithm(reference_name, graph, source)
    # BFS hop counts are a different task; compare them only to themselves.
    rows: list[Observation] = []
    for name in names:
        distances = run_algorithm(name, graph, source)
        if name == "bfs":
            correct = results_match(distances, bfs_shortest_paths(graph.adj, source))
        elif reference_name == "bfs":
            correct = True
        else:
            correct = results_match(distances, reference)
        samples = measure_runtime_ms(lambda n=name: run_algorithm(n, graph, source), repeats=repeats)
        peak = measure_peak_kb(lambda n=name: run_algorithm(n, graph, source))
        for i, sample in enumerate(samples):
            rows.append(
                Observation(
                    graph_id=graph.graph_id,
                    graph_family=graph.family,
                    seed=graph.seed,
                    vertices=graph.n,
                    edges=graph.m_undirected,
                    algorithm=name,
                    run=i,
                    runtime_ms=sample,
                    peak_memory_kb=peak,
                    correct=correct,
                )
            )
    return rows

"""Feasible-set selectors: fixed, rule, predict-then-select, classifier, oracle."""
from __future__ import annotations

from algoselect.algorithms import ALGORITHMS
from algoselect.benchmark import feasible
from algoselect.features import vector
from algoselect.graphs import GraphInstance


def rule_based(graph: GraphInstance, feats: dict[str, float]) -> str:
    """Hand rule from the usual complexity regimes. Documented, not learned."""
    names = feasible(graph)
    if "bfs" in names and graph.unweighted:
        return "bfs"
    if feats["weight_min"] < 0:
        return "bellman_ford"
    if feats["density"] >= 0.2 and "dijkstra_array" in names:
        return "dijkstra_array"
    if feats["weight_bound"] <= 15 and feats["density"] < 0.08 and "dial" in names:
        return "dial"
    if "dijkstra_heap" in names:
        return "dijkstra_heap"
    return names[0]


def fixed_baseline(graph: GraphInstance) -> str:
    names = feasible(graph)
    if "dijkstra_heap" in names:
        return "dijkstra_heap"
    return names[0]


def oracle(runtime: dict[str, float]) -> str:
    return min(runtime, key=runtime.get)


def design_row(feats: dict[str, float], names: tuple[str, ...], algorithm: str) -> list[float]:
    base = vector(feats, names)
    return base + [1.0 if algorithm == name else 0.0 for name in ALGORITHMS]


def predict_then_select(model, graph: GraphInstance, feats: dict[str, float], names: tuple[str, ...]) -> str:
    candidates = feasible(graph)
    rows = [design_row(feats, names, algo) for algo in candidates]
    preds = model.predict(rows)
    return candidates[min(range(len(candidates)), key=lambda i: preds[i])]


def memory_constrained(time_model, mem_model, graph, feats, names, reference: str = "dijkstra_heap") -> str:
    """Fastest predicted algorithm whose predicted memory does not exceed the reference algorithm."""
    candidates = feasible(graph)
    t_rows = [design_row(feats, names, algo) for algo in candidates]
    times = time_model.predict(t_rows)
    mems = mem_model.predict(t_rows)
    if reference in candidates:
        cap = mems[candidates.index(reference)]
    else:
        cap = max(mems)
    allowed = [i for i, mem in enumerate(mems) if mem <= cap * 1.05 + 1e-9]
    pool = allowed or list(range(len(candidates)))
    return candidates[min(pool, key=lambda i: times[i])]

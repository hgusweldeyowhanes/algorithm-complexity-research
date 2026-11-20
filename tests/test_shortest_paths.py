from math import inf

import pytest

from algoselect.algorithms import bellman_ford, bfs_shortest_paths, dial, dijkstra_array, dijkstra_heap
from algoselect.benchmark import feasible, results_match, run_algorithm
from algoselect.features import FULL, extract
from algoselect.graphs import erdos_renyi, grid_graph, portfolio
from algoselect.selection import rule_based


def _path_graph():
    # 0-1-2 with weights 2, 3. Undirected.
    adj = [[(1, 2)], [(0, 2), (2, 3)], [(1, 3)]]
    edges = [(0, 1, 2), (1, 0, 2), (1, 2, 3), (2, 1, 3)]
    return adj, edges


def test_weighted_algorithms_agree_on_a_path():
    adj, edges = _path_graph()
    heap = dijkstra_heap(adj, 0)
    array = dijkstra_array(adj, 0)
    ford = bellman_ford(3, edges, 0)
    buckets = dial(adj, 0, 3)
    assert heap == [0, 2, 5]
    assert results_match(array, heap)
    assert results_match(ford, heap)
    assert results_match(buckets, heap)


def test_bfs_counts_hops_not_weights():
    adj, _edges = _path_graph()
    assert bfs_shortest_paths(adj, 0) == [0, 1, 2]


def test_negative_cycle_and_dijkstra_guard():
    edges = [(0, 1, 1), (1, 2, -2), (2, 0, -2)]
    with pytest.raises(ValueError):
        bellman_ford(3, edges, 0)
    adj = [[(1, -1)], [(0, -1)]]
    with pytest.raises(ValueError):
        dijkstra_heap(adj, 0)


def test_generated_graph_is_correct_and_features_are_structural():
    graph = erdos_renyi(25, 4, seed=3)
    assert "dijkstra_heap" in feasible(graph)
    assert "bfs" not in feasible(graph)
    reference = run_algorithm("dijkstra_heap", graph)
    for name in ("dijkstra_array", "bellman_ford", "dial"):
        assert results_match(run_algorithm(name, graph), reference)
    feats = extract(graph)
    assert set(FULL) <= set(feats)
    assert "runtime_ms" not in feats
    assert feats["vertices"] == 25


def test_unweighted_admits_bfs_and_grid_connects():
    graph = erdos_renyi(12, 3, seed=1, unweighted=True)
    assert "bfs" in feasible(graph)
    assert results_match(run_algorithm("bfs", graph), run_algorithm("dijkstra_heap", graph))
    grid = grid_graph(9, seed=1)
    assert grid.m_undirected >= 8
    assert all(math_dist < inf for math_dist in run_algorithm("dijkstra_heap", grid))


def test_portfolio_covers_selection_regimes():
    graphs = portfolio(seed=2)
    assert max(g.n for g in graphs) >= 400
    dense = [g for g in graphs if extract(g)["density"] >= 0.12]
    unweighted = [g for g in graphs if g.unweighted]
    low_c = [g for g in graphs if not g.unweighted and extract(g)["weight_bound"] <= 2]
    assert dense and unweighted and low_c
    assert rule_based(unweighted[0], extract(unweighted[0])) == "bfs"
    assert rule_based(low_c[0], extract(low_c[0])) == "dial"

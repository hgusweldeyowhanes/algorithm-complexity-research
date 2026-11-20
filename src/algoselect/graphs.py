"""Synthetic graph families and a small edge-list loader. Undirected integer weights."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class GraphInstance:
    graph_id: str
    family: str
    seed: int
    n: int
    adj: list[list[tuple[int, int]]]
    edges: list[tuple[int, int, int]]
    unweighted: bool = False
    notes: str = ""
    meta: dict[str, float] = field(default_factory=dict)

    @property
    def m_directed(self) -> int:
        return len(self.edges)

    @property
    def m_undirected(self) -> int:
        return len(self.edges) // 2


def _finalize(
    n: int,
    undirected: dict[tuple[int, int], int],
    *,
    graph_id: str,
    family: str,
    seed: int,
    unweighted: bool,
) -> GraphInstance:
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    edges: list[tuple[int, int, int]] = []
    for (u, v), weight in undirected.items():
        adj[u].append((v, weight))
        adj[v].append((u, weight))
        edges.append((u, v, weight))
        edges.append((v, u, weight))
    return GraphInstance(
        graph_id=graph_id,
        family=family,
        seed=seed,
        n=n,
        adj=adj,
        edges=edges,
        unweighted=unweighted,
    )


def _add(edges: dict[tuple[int, int], int], u: int, v: int, weight: int) -> None:
    if u == v:
        return
    a, b = (u, v) if u < v else (v, u)
    edges[(a, b)] = weight


def erdos_renyi(n: int, average_degree: float, *, seed: int, weight_hi: int = 10, unweighted: bool = False) -> GraphInstance:
    rng = random.Random(seed)
    p = 0.0 if n <= 1 else min(1.0, average_degree / (n - 1))
    edges: dict[tuple[int, int], int] = {}
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() <= p:
                w = 1 if unweighted else rng.randint(1, weight_hi)
                _add(edges, u, v, w)
    return _finalize(
        n, edges, graph_id=f"er-{n}-{average_degree}-{seed}-{int(unweighted)}",
        family="erdos_renyi", seed=seed, unweighted=unweighted,
    )


def barabasi_albert(n: int, average_degree: float, *, seed: int, weight_hi: int = 10) -> GraphInstance:
    rng = random.Random(seed)
    m = max(1, int(average_degree // 2))
    m = min(m, max(1, n - 1))
    edges: dict[tuple[int, int], int] = {}
    targets = list(range(m + 1))
    for u in range(m + 1):
        for v in range(u + 1, m + 1):
            _add(edges, u, v, rng.randint(1, weight_hi))
    degree_ends: list[int] = []
    for u, v in list(edges):
        degree_ends.append(u)
        degree_ends.append(v)
    for new in range(m + 1, n):
        chosen: set[int] = set()
        while len(chosen) < min(m, new):
            chosen.add(degree_ends[rng.randrange(len(degree_ends))] if degree_ends else rng.randrange(new))
        for v in chosen:
            _add(edges, new, v, rng.randint(1, weight_hi))
            degree_ends.append(new)
            degree_ends.append(v)
    return _finalize(n, edges, graph_id=f"ba-{n}-{m}-{seed}", family="barabasi_albert", seed=seed, unweighted=False)


def watts_strogatz(n: int, average_degree: float, *, seed: int, rewire: float = 0.1, weight_hi: int = 10) -> GraphInstance:
    rng = random.Random(seed)
    k = int(average_degree)
    if k % 2:
        k -= 1
    k = max(2, min(k, n - 1 if (n - 1) % 2 == 0 else n - 2))
    edges: dict[tuple[int, int], int] = {}
    for u in range(n):
        for step in range(1, k // 2 + 1):
            v = (u + step) % n
            weight = rng.randint(1, weight_hi)
            if rng.random() < rewire:
                placed = False
                for _try in range(16):
                    dest = rng.randrange(n)
                    key = (u, dest) if u < dest else (dest, u)
                    if dest != u and key not in edges:
                        _add(edges, u, dest, weight)
                        placed = True
                        break
                if not placed:
                    _add(edges, u, v, weight)
            else:
                _add(edges, u, v, weight)
    return _finalize(n, edges, graph_id=f"ws-{n}-{k}-{seed}", family="watts_strogatz", seed=seed, unweighted=False)


def random_geometric(n: int, average_degree: float, *, seed: int, weight_hi: int = 10) -> GraphInstance:
    rng = random.Random(seed)
    points = [(rng.random(), rng.random()) for _ in range(n)]
    radius = math.sqrt(max(average_degree, 1.0) / (math.pi * max(n, 1)))
    radius = min(0.75, max(radius, 0.05))
    edges: dict[tuple[int, int], int] = {}
    for u in range(n):
        for v in range(u + 1, n):
            dx = points[u][0] - points[v][0]
            dy = points[u][1] - points[v][1]
            if dx * dx + dy * dy <= radius * radius:
                _add(edges, u, v, rng.randint(1, weight_hi))
    return _finalize(n, edges, graph_id=f"geo-{n}-{average_degree}-{seed}", family="geometric", seed=seed, unweighted=False)


def grid_graph(n: int, *, seed: int, weight_hi: int = 10) -> GraphInstance:
    rng = random.Random(seed)
    side = max(1, math.ceil(math.sqrt(n)))
    edges: dict[tuple[int, int], int] = {}
    for i in range(n):
        row, col = divmod(i, side)
        right = i + 1
        if col + 1 < side and right < n and right // side == row:
            _add(edges, i, right, rng.randint(1, weight_hi))
        down = i + side
        if down < n:
            _add(edges, i, down, rng.randint(1, weight_hi))
    return _finalize(n, edges, graph_id=f"grid-{n}-{seed}", family="grid", seed=seed, unweighted=False)


def load_edge_list(path: str, *, graph_id: str = "edge-list") -> GraphInstance:
    """Undirected edge list: `u v w` per line. Nodes are renumbered to 0..n-1."""
    raw: list[tuple[int, int, int]] = []
    ids: set[int] = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            u, v = int(parts[0]), int(parts[1])
            w = int(parts[2]) if len(parts) > 2 else 1
            raw.append((u, v, w))
            ids.add(u)
            ids.add(v)
    mapping = {old: i for i, old in enumerate(sorted(ids))}
    edges: dict[tuple[int, int], int] = {}
    for u, v, w in raw:
        _add(edges, mapping[u], mapping[v], w)
    graph = _finalize(
        len(mapping), edges, graph_id=graph_id, family="edge_list", seed=0,
        unweighted=all(w == 1 for w in edges.values()),
    )
    graph.notes = path
    return graph


def portfolio(*, seed: int = 1) -> list[GraphInstance]:
    """Local-PC portfolio sized so algorithms diverge: sparse, dense, low-C, unweighted."""
    graphs: list[GraphInstance] = []
    # Sparse / mid weight — heap often wins.
    for n in (80, 200, 400):
        for degree in (4, 8):
            graphs.append(erdos_renyi(n, degree, seed=seed, weight_hi=50))
            graphs.append(barabasi_albert(n, degree, seed=seed, weight_hi=50))
            graphs.append(watts_strogatz(n, degree, seed=seed, weight_hi=50))
            graphs.append(random_geometric(n, degree, seed=seed, weight_hi=50))
        graphs.append(grid_graph(81 if n == 80 else n, seed=seed, weight_hi=50))
    # Dial-friendly: tiny integer bound, sparse, larger n (circular buckets O(E+VC)).
    for n in (120, 250, 500):
        graphs.append(erdos_renyi(n, 3, seed=seed + 7, weight_hi=1))
        graphs.append(erdos_renyi(n, 4, seed=seed + 8, weight_hi=2))
        graphs.append(barabasi_albert(n, 3, seed=seed + 7, weight_hi=1))
        graphs.append(grid_graph(n, seed=seed + 7, weight_hi=1))
    # Unweighted — BFS should beat heap Dijkstra.
    for n in (80, 160, 320, 480):
        graphs.append(erdos_renyi(n, 4, seed=seed + 3, unweighted=True))
        graphs.append(erdos_renyi(n, 6, seed=seed + 4, unweighted=True))
        ba = barabasi_albert(n, 4, seed=seed + 5, weight_hi=1)
        ba.unweighted = True
        ba.graph_id = f"{ba.graph_id}-uw"
        graphs.append(ba)
    # Dense — array Dijkstra asymptotic regime (skip huge BF by keeping n<=220).
    for n in (120, 180, 220):
        degree = max(20.0, 0.55 * (n - 1))
        graphs.append(erdos_renyi(n, degree, seed=seed, weight_hi=10))
        graphs.append(erdos_renyi(n, degree, seed=seed + 11, weight_hi=10))
    return graphs

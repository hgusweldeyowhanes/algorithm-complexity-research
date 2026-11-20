"""Shortest-path algorithms. Each returns distances from a single source."""
from __future__ import annotations

import heapq
from math import inf


def bfs_shortest_paths(adj: list[list[tuple[int, int]]], source: int) -> list[float]:
    """Unweighted hop distance. Ignores edge weights. O(V+E)."""
    n = len(adj)
    if not 0 <= source < n:
        raise ValueError("Invalid source node")
    distances = [inf] * n
    distances[source] = 0
    queue = [source]
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        for v, _weight in adj[u]:
            if distances[v] == inf:
                distances[v] = distances[u] + 1
                queue.append(v)
    return distances


def dijkstra_heap(adj: list[list[tuple[int, int]]], source: int) -> list[float]:
    """Binary-heap Dijkstra. Nonnegative weights. O((V+E) log V)."""
    n = len(adj)
    if not 0 <= source < n:
        raise ValueError("Invalid source node")
    distances = [inf] * n
    distances[source] = 0
    priority_queue = [(0, source)]
    while priority_queue:
        current_distance, u = heapq.heappop(priority_queue)
        if current_distance != distances[u]:
            continue
        for v, weight in adj[u]:
            if weight < 0:
                raise ValueError("Dijkstra cannot handle negative weights")
            new_distance = current_distance + weight
            if new_distance < distances[v]:
                distances[v] = new_distance
                heapq.heappush(priority_queue, (new_distance, v))
    return distances


def dijkstra_array(adj: list[list[tuple[int, int]]], source: int) -> list[float]:
    """Array-scan Dijkstra. Nonnegative weights. O(V^2 + E)."""
    n = len(adj)
    if not 0 <= source < n:
        raise ValueError("Invalid source node")
    distances = [inf] * n
    visited = [False] * n
    distances[source] = 0
    for _ in range(n):
        u = None
        minimum_distance = inf
        for node in range(n):
            if not visited[node] and distances[node] < minimum_distance:
                minimum_distance = distances[node]
                u = node
        if u is None:
            break
        visited[u] = True
        for v, weight in adj[u]:
            if weight < 0:
                raise ValueError("Dijkstra cannot handle negative weights")
            if visited[v]:
                continue
            new_distance = distances[u] + weight
            if new_distance < distances[v]:
                distances[v] = new_distance
    return distances


def bellman_ford(
    n: int,
    edges: list[tuple[int, int, int]],
    source: int,
) -> list[float]:
    """O(VE). Raises if a negative cycle is reachable from the source."""
    if not 0 <= source < n:
        raise ValueError("Invalid source node")
    distances = [inf] * n
    distances[source] = 0
    for _ in range(n - 1):
        updated = False
        for u, v, weight in edges:
            if distances[u] == inf:
                continue
            new_distance = distances[u] + weight
            if new_distance < distances[v]:
                distances[v] = new_distance
                updated = True
        if not updated:
            break
    for u, v, weight in edges:
        if distances[u] != inf and distances[u] + weight < distances[v]:
            raise ValueError("Graph contains a reachable negative cycle")
    return distances


def dial(adj: list[list[tuple[int, int]]], source: int, max_weight: int) -> list[float]:
    """
    Dial's bucket Dijkstra for nonnegative integer weights bounded by C.
    Circular buckets of size C+1; time O(E + V*C).
    """
    n = len(adj)
    if not 0 <= source < n:
        raise ValueError("Invalid source node")
    if max_weight < 0:
        raise ValueError("Dial requires nonnegative weights")
    bucket_count = max(max_weight, 0) + 1
    distances = [inf] * n
    distances[source] = 0
    buckets: list[list[int]] = [[] for _ in range(bucket_count)]
    buckets[0].append(source)
    distance = 0
    empty_streak = 0
    while empty_streak <= max_weight:
        bucket = distance % bucket_count
        if not buckets[bucket]:
            empty_streak += 1
            distance += 1
            continue
        empty_streak = 0
        while buckets[bucket]:
            u = buckets[bucket].pop()
            if distances[u] != distance:
                continue
            for v, weight in adj[u]:
                if weight < 0:
                    raise ValueError("Dial cannot handle negative weights")
                new_distance = distances[u] + weight
                if new_distance < distances[v]:
                    distances[v] = new_distance
                    buckets[int(new_distance) % bucket_count].append(v)
        distance += 1
    return distances


ALGORITHMS = ("bfs", "dijkstra_heap", "dijkstra_array", "bellman_ford", "dial")

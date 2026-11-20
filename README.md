# Algorithm complexity research

Learning-based **shortest-path algorithm selection**, from `algorithm_complexity_research_handbook.md`.

The system generates graphs, keeps only algorithms that are valid for that graph, measures runtime and memory, and compares a learned selector with a fixed algorithm, a hand-written rule, and an oracle.

| Algorithm | When it is allowed | Complexity used as a hint |
|-----------|--------------------|---------------------------|
| BFS | Every weight is 1 | O(V+E) hops |
| Dijkstra, binary heap | Nonnegative weights | O((V+E) log V) |
| Dijkstra, array scan | Nonnegative weights | O(V²+E) |
| Dial | Nonnegative integer weights | O(E+VC) |
| Bellman–Ford | Any weights; rejects a reachable negative cycle | O(VE) |

## Run

```bash
cd algorithm-complexity-research
PYTHONPATH=src python -m algoselect --seed 1
python -m pytest -q
```

`results/PAPER.md` is the measured write-up (prediction, selection regret, ablation, threats). `results/observations.csv` keeps every timing repetition.

Graph families are Erdős–Rényi, Barabási–Albert, Watts–Strogatz, random geometric, and grid. `data/raw/sample_edge_list.txt` is a tiny stand-in for an external edge list. Network Repository and DIMACS graphs are not bundled.

## Layout

```
src/algoselect/
  algorithms.py   shortest paths
  graphs.py       generators and edge-list loader
  features.py     structure and weights (no weighted all-pairs)
  benchmark.py    feasibility, timing, tracemalloc
  learn.py        median, linear, forest, decision tree
  selection.py    fixed, rule, predict-then-select, memory cap, oracle
  study.py        experiments and PAPER.md
```

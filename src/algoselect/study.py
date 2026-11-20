"""Experiments 1–5 and the written research note."""
from __future__ import annotations

import csv
import os
import platform
import random
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from algoselect.benchmark import Observation, benchmark_graph, feasible
from algoselect.features import BASIC_COUNTS, FULL, STRUCTURAL, WEIGHTS, timed_extract, vector
from algoselect.graphs import GraphInstance, load_edge_list, portfolio
from algoselect.learn import ForestLog, LinearLog, MedianBaseline, Score, TreeClassifier, loglog_slope, score
from algoselect.selection import (
    design_row,
    fixed_baseline,
    memory_constrained,
    oracle,
    predict_then_select,
    rule_based,
)


@dataclass
class Case:
    graph: GraphInstance
    feats: dict[str, float]
    feature_ms: float
    runtime: dict[str, float]
    memory: dict[str, float]
    correct: dict[str, bool]
    rows: list[Observation] = field(default_factory=list)


def build_cases(graphs: list[GraphInstance], repeats: int = 2) -> list[Case]:
    cases: list[Case] = []
    for graph in graphs:
        feats, feature_ms = timed_extract(graph)
        rows = benchmark_graph(graph, repeats=repeats)
        grouped: dict[str, list[Observation]] = defaultdict(list)
        for row in rows:
            grouped[row.algorithm].append(row)
        runtime = {name: statistics.median(r.runtime_ms for r in items) for name, items in grouped.items()}
        memory = {name: statistics.median(r.peak_memory_kb for r in items) for name, items in grouped.items()}
        correct = {name: all(r.correct for r in items) for name, items in grouped.items()}
        cases.append(Case(graph, feats, feature_ms, runtime, memory, correct, rows))
    return cases


def _xy(cases: list[Case], names: tuple[str, ...], target: str):
    X, y = [], []
    for case in cases:
        for algo, value in (case.runtime if target == "runtime" else case.memory).items():
            X.append(design_row(case.feats, names, algo))
            y.append(value)
    return X, y


def _fit_trio(train: list[Case], names: tuple[str, ...], target: str):
    X, y = _xy(train, names, target)
    models = [MedianBaseline().fit(X, y), LinearLog().fit(X, y), ForestLog(seed=1).fit(X, y)]
    return models


def _eval_models(models, test: list[Case], names, target: str) -> list[Score]:
    X, y = _xy(test, names, target)
    return [score(m.name, target, y, m.predict(X)) for m in models]


def _choice_time(case: Case, algo: str) -> float:
    return case.runtime[algo]


def _summarize_selector(
    name: str,
    pairs: list[tuple[Case, str, float]],
    *,
    include_overhead: bool,
) -> dict[str, float]:
    """pairs: case, chosen algo, inference ms. Overhead is feature extraction plus inference."""
    base_times, chosen, oracle_times, e2e, regrets = [], [], [], [], []
    for case, algo, infer_ms in pairs:
        base = _choice_time(case, fixed_baseline(case.graph))
        got = _choice_time(case, algo)
        best = _choice_time(case, oracle(case.runtime))
        base_times.append(base)
        chosen.append(got)
        oracle_times.append(best)
        extra = (case.feature_ms + infer_ms) if include_overhead else 0.0
        e2e.append(extra + got)
        regrets.append(got / best if best > 0 else 1.0)
    return {
        "selector": name,
        "graphs": float(len(pairs)),
        "median_runtime_ms": statistics.median(chosen) if chosen else 0.0,
        "median_baseline_ms": statistics.median(base_times) if base_times else 0.0,
        "speedup": (statistics.median(base_times) / statistics.median(chosen)) if chosen and statistics.median(chosen) else 0.0,
        "median_regret": statistics.median(regrets) if regrets else 0.0,
        "median_e2e_ms": statistics.median(e2e) if e2e else 0.0,
        "oracle_median_ms": statistics.median(oracle_times) if oracle_times else 0.0,
    }


def _time_predict(model, rows) -> tuple[list, float]:
    t0 = time.perf_counter()
    pred = model.predict(rows)
    return pred, (time.perf_counter() - t0) * 1000.0


def evaluate_selectors(train: list[Case], test: list[Case], feature_names: tuple[str, ...]) -> list[dict[str, float]]:
    time_models = _fit_trio(train, feature_names, "runtime")
    mem_models = _fit_trio(train, feature_names, "memory")
    linear_time = time_models[1]
    linear_mem = mem_models[1]
    Xc, yc = [], []
    for case in train:
        Xc.append(vector(case.feats, feature_names))
        yc.append(oracle(case.runtime))
    clf = TreeClassifier(depth=4, seed=2).fit(Xc, yc)

    buckets = {
        "fixed_dijkstra_heap": [],
        "rule_based": [],
        "predict_then_select": [],
        "decision_tree": [],
        "memory_constrained": [],
        "oracle": [],
    }
    for case in test:
        buckets["fixed_dijkstra_heap"].append((case, fixed_baseline(case.graph), 0.0))
        buckets["rule_based"].append((case, rule_based(case.graph, case.feats), 0.0))
        t0 = time.perf_counter()
        chosen = predict_then_select(linear_time, case.graph, case.feats, feature_names)
        infer = (time.perf_counter() - t0) * 1000.0
        buckets["predict_then_select"].append((case, chosen, infer))
        t0 = time.perf_counter()
        label = clf.predict([vector(case.feats, feature_names)])[0]
        infer = (time.perf_counter() - t0) * 1000.0
        if label not in feasible(case.graph):
            label = rule_based(case.graph, case.feats)
        buckets["decision_tree"].append((case, label, infer))
        t0 = time.perf_counter()
        constrained = memory_constrained(linear_time, linear_mem, case.graph, case.feats, feature_names)
        infer = (time.perf_counter() - t0) * 1000.0
        buckets["memory_constrained"].append((case, constrained, infer))
        buckets["oracle"].append((case, oracle(case.runtime), 0.0))
    no_overhead = {"fixed_dijkstra_heap", "oracle"}
    return [
        _summarize_selector(name, pairs, include_overhead=name not in no_overhead)
        for name, pairs in buckets.items()
    ]


def growth_table(cases: list[Case]) -> list[dict]:
    focused = [
        c for c in cases
        if c.graph.family == "erdos_renyi" and not c.graph.unweighted and abs(c.feats["average_degree"] - 4) < 1.5
    ]
    by_algo: dict[str, list[Case]] = defaultdict(list)
    for case in focused:
        for algo in case.runtime:
            by_algo[algo].append(case)
    rows = []
    theory = {
        "bfs": 1.0,
        "dijkstra_heap": 1.0,
        "dijkstra_array": 2.0,
        "bellman_ford": 2.0,
        "dial": 1.0,
    }
    for algo, items in by_algo.items():
        # one point per size: median across seeds
        by_n: dict[int, list[float]] = defaultdict(list)
        for case in items:
            by_n[case.graph.n].append(case.runtime[algo])
        sizes = sorted(by_n)
        times = [statistics.median(by_n[n]) for n in sizes]
        rows.append({
            "algorithm": algo,
            "sizes": sizes,
            "times": times,
            "loglog_slope": loglog_slope([float(n) for n in sizes], times),
            "theory_hint": theory.get(algo, 0.0),
        })
    return rows


def ablation(train: list[Case], test: list[Case]) -> list[dict]:
    sets = {
        "counts_only": BASIC_COUNTS,
        "structural": STRUCTURAL,
        "structural_and_weights": STRUCTURAL + WEIGHTS,
    }
    rows = []
    for name, names in sets.items():
        model = LinearLog().fit(*_xy(train, names, "runtime"))
        X, y = _xy(test, names, "runtime")
        pred = model.predict(X)
        sc = score(name, "runtime", y, pred)
        summary = evaluate_selectors(train, test, names)
        pts = next(s for s in summary if s["selector"] == "predict_then_select")
        rows.append({
            "features": name,
            "mae_ms": sc.mae,
            "r2": sc.r2,
            "regret": pts["median_regret"],
            "feature_count": len(names) + 5,
        })
    return rows


def permutation_importance(train: list[Case], test: list[Case]) -> list[tuple[str, float]]:
    names = FULL
    model = LinearLog().fit(*_xy(train, names, "runtime"))
    X, y = _xy(test, names, "runtime")
    base = score("base", "runtime", y, model.predict(X)).mae
    rng = random.Random(0)
    ranked = []
    for j, name in enumerate(names):
        shuffled = [row[:] for row in X]
        col = [row[j] for row in shuffled]
        rng.shuffle(col)
        for row, value in zip(shuffled, col):
            row[j] = value
        mae = score("perm", "runtime", y, model.predict(shuffled)).mae
        ranked.append((name, mae - base))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:8]


def split_cases(cases: list[Case], seed: int = 1):
    rng = random.Random(seed)
    small = [c for c in cases if c.graph.family != "geometric" and c.graph.family != "edge_list" and c.graph.n < 70]
    rng.shuffle(small)
    hold = max(1, len(small) // 4)
    instance_test = small[:hold]
    train = small[hold:]
    size_test = [c for c in cases if c.graph.family not in {"geometric", "edge_list"} and c.graph.n >= 70]
    family_test = [c for c in cases if c.graph.family == "geometric"]
    transfer = [c for c in cases if c.graph.family == "edge_list"]
    return train, instance_test, size_test, family_test, transfer


def write_outputs(study: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / "observations.csv"
    with raw.open("w", newline="", encoding="utf-8") as fh:
        fields = list(Observation.__dataclass_fields__.keys())
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for case in study["cases"]:
            for row in case.rows:
                writer.writerow({k: getattr(row, k) for k in fields})

    def score_table(scores: list[Score]) -> str:
        lines = ["| Model | Target | MAE | RMSE | R² |", "|-------|--------|-----|------|----|"]
        for s in scores:
            lines.append(f"| {s.name} | {s.target} | {s.mae:.3f} | {s.rmse:.3f} | {s.r2:.3f} |")
        return "\n".join(lines)

    def sel_table(rows: list[dict]) -> str:
        lines = [
            "| Selector | Graphs | Median ms | Speedup vs heap | Regret vs oracle | End-to-end ms |",
            "|----------|--------|-----------|-----------------|------------------|---------------|",
        ]
        for row in rows:
            lines.append(
                f"| {row['selector']} | {int(row['graphs'])} | {row['median_runtime_ms']:.3f} | "
                f"{row['speedup']:.2f} | {row['median_regret']:.2f} | {row['median_e2e_ms']:.3f} |"
            )
        return "\n".join(lines)

    growth_lines = ["| Algorithm | Sizes | Median ms | log-log slope | Polynomial hint |", "|-----------|-------|-----------|---------------|-----------------|"]
    for row in study["growth"]:
        growth_lines.append(
            f"| {row['algorithm']} | {row['sizes']} | {[round(t, 3) for t in row['times']]} | "
            f"{row['loglog_slope']:.2f} | {row['theory_hint']:.1f} |"
        )
    imp_lines = ["| Feature | MAE increase when shuffled |", "|---------|----------------------------|"]
    for name, delta in study["importance"]:
        imp_lines.append(f"| {name} | {delta:.4f} |")
    abl_lines = ["| Feature set | Runtime MAE (ms) | R² | Selection regret |", "|-------------|------------------|----|------------------|"]
    for row in study["ablation"]:
        abl_lines.append(f"| {row['features']} | {row['mae_ms']:.3f} | {row['r2']:.3f} | {row['regret']:.2f} |")

    inst = study["instance_scores"]
    best_runtime = min((s for s in inst if s.target == "runtime"), key=lambda s: s.mae)
    mean_runtime = next(s for s in inst if s.name == "median_baseline" and s.target == "runtime")
    h1 = best_runtime.r2 > 0.3 and best_runtime.mae < mean_runtime.mae
    counts = next(r for r in study["ablation"] if r["features"] == "counts_only")
    full = next(r for r in study["ablation"] if r["features"] == "structural_and_weights")
    h2 = full["mae_ms"] < counts["mae_ms"]
    inst_sel = study["instance_selection"]
    fixed = next(r for r in inst_sel if r["selector"] == "fixed_dijkstra_heap")
    learned = next(r for r in inst_sel if r["selector"] == "predict_then_select")
    h3 = learned["median_runtime_ms"] < fixed["median_runtime_ms"]
    fam = study["family_selection"]
    fam_learned = next(r for r in fam if r["selector"] == "predict_then_select")
    h4 = fam_learned["median_regret"] > learned["median_regret"]

    text = f"""# Learning-based algorithm selection for shortest paths

Measured study for the handbook *Machine Learning-Based Algorithmic Complexity Prediction and Algorithm Selection*. This is one local run, not a published result.

- Python {sys.version.split()[0]} on {platform.system()} {platform.release()}
- CPU count: {os.cpu_count()}
- Graphs: Erdős–Rényi, Barabási–Albert, Watts–Strogatz, random geometric, grid, one unweighted ER series, and `data/raw/sample_edge_list.txt`
- Feasible set: BFS only when every weight is 1; heap Dijkstra, array Dijkstra, and Dial on nonnegative integer weights; Bellman–Ford on every graph in this set
- Splits are by graph instance. Repeated timings of the same graph stay together.

## Abstract

The question is whether graph features can predict the runtime and memory of feasible shortest-path algorithms well enough to choose one that beats a fixed Dijkstra implementation, after feature-extraction cost. On this machine, the best runtime model {"beat" if h1 else "did not beat"} a median baseline on unseen instances from familiar families ({best_runtime.name}, R² {best_runtime.r2:.2f}). Predict-then-select {"reduced" if h3 else "did not reduce"} median runtime versus always using binary-heap Dijkstra. Held-out geometric graphs were {"harder" if h4 else "not harder"} than new instances of familiar families, measured by selection regret.

## 1. Algorithms and complexity evidence

Log-log slopes are fit to median runtime against vertex count on Erdős–Rényi graphs with average degree near 4. A slope near 1 is consistent with roughly linear growth in V over this range when E grows with V. A slope near 2 is consistent with quadratic growth. Two or three sizes do not prove a Big-O bound.

{chr(10).join(growth_lines)}

## 2. Prediction accuracy

Instance split: train and test are different graphs from the same non-geometric families and the small size band.

{score_table(study['instance_scores'])}

Size shift: models trained on graphs with fewer than 70 vertices, tested on larger familiar-family graphs.

{score_table(study['size_scores'])}

**RQ1 / H1.** Useful accuracy means beating the median baseline with R² above 0.3 on the instance split. This run: {"supported" if h1 else "not supported"} (`{best_runtime.name}` MAE {best_runtime.mae:.3f} ms vs median baseline {mean_runtime.mae:.3f} ms).

**RQ2.** Memory scores are the rows whose target is `memory` in the same tables. Tracemalloc peak is Python-level allocation, not process RSS.

**RQ3.** Shuffling a feature on the instance-split test and re-scoring the linear runtime model. Larger MAE increase means the model was using that column.

{chr(10).join(imp_lines)}

## 3. Algorithm selection

Speedup is median fixed-heap runtime divided by median selected runtime. Regret is selected runtime divided by the oracle runtime for that graph; 1.0 means the selector matched the fastest measured algorithm. End-to-end time adds feature extraction and inference.

### Unseen instances, familiar families

{sel_table(study['instance_selection'])}

### Larger graphs, familiar families

{sel_table(study['size_selection'])}

### Held-out geometric family

{sel_table(study['family_selection'])}

### Edge-list transfer

{sel_table(study['transfer_selection'])}

**RQ4 / H3.** Predict-then-select versus always using heap Dijkstra on the instance split: {"supported" if h3 else "not supported"} (selected median {learned['median_runtime_ms']:.3f} ms, fixed {fixed['median_runtime_ms']:.3f} ms, regret {learned['median_regret']:.2f}).

**RQ5 / H4.** Familiar-instance regret {learned['median_regret']:.2f} versus geometric-family regret {fam_learned['median_regret']:.2f}. {"Supported: the new family is harder." if h4 else "Not supported on this run: the held-out family was not harder by regret."}

**RQ6.** Compare end-to-end milliseconds with the fixed selector, which pays no feature cost. If end-to-end exceeds the fixed runtime, prediction overhead erased the scheduling gain on these small graphs.

The rule is: BFS on unweighted graphs; array Dijkstra when density is at least 0.2; Dial when the weight bound is at most 15 and density is under 0.08; otherwise heap Dijkstra. The memory-constrained selector refuses an algorithm whose predicted memory exceeds heap Dijkstra's predicted memory.

## 4. Ablation

{chr(10).join(abl_lines)}

**H2.** Structural and weight features versus vertex/edge counts only: {"supported" if h2 else "not supported"} (MAE {full['mae_ms']:.3f} ms vs {counts['mae_ms']:.3f} ms).

## 5. Threats to validity

- Graph sizes stay in the tens or low hundreds of vertices so Bellman–Ford and array Dijkstra finish on one PC. Slopes can change at DIMACS scale.
- Two repetitions, not 30. Medians are stable only for large gaps.
- The sample edge list is a toy file in this repo, not Network Repository or the DIMACS challenge.
- BFS is a different problem (hop count). It enters the feasible set only for unit weights, where hop count and weighted distance coincide.
- Feature extraction uses one BFS for depth and a clustering pass. It does not compute all-pairs weighted distances.
- No second machine was available.

## 6. Conclusion

The portfolio, feasibility filter, and oracle comparison are in place. Whether selection is worth its overhead depends on the gap between algorithms. On graphs this small, heap Dijkstra is often already close to the oracle, so a learned selector has little room to improve and feature extraction can cost more than it saves. The same pipeline is the one to rerun when a sampled real network is dropped in as an edge list.
"""
    path = out_dir / "PAPER.md"
    path.write_text(text, encoding="utf-8")
    return path


def execute(out_dir: Path, seed: int = 1) -> Path:
    graphs = portfolio(seed=seed) + portfolio(seed=seed + 1)
    sample = Path(__file__).resolve().parents[2] / "data" / "raw" / "sample_edge_list.txt"
    if sample.exists():
        graphs.append(load_edge_list(str(sample), graph_id="sample-edge-list"))
    cases = build_cases(graphs, repeats=2)
    bad = [c.graph.graph_id for c in cases if not all(c.correct.values())]
    if bad:
        raise RuntimeError(f"correctness failed: {bad}")
    train, instance_test, size_test, family_test, transfer = split_cases(cases, seed=seed)
    study = {
        "cases": cases,
        "growth": growth_table(cases),
        "instance_scores": _eval_models(_fit_trio(train, FULL, "runtime"), instance_test, FULL, "runtime")
        + _eval_models(_fit_trio(train, FULL, "memory"), instance_test, FULL, "memory"),
        "size_scores": _eval_models(_fit_trio(train, FULL, "runtime"), size_test, FULL, "runtime")
        + _eval_models(_fit_trio(train, FULL, "memory"), size_test, FULL, "memory"),
        "instance_selection": evaluate_selectors(train, instance_test, FULL),
        "size_selection": evaluate_selectors(train, size_test, FULL),
        "family_selection": evaluate_selectors(train, family_test, FULL),
        "transfer_selection": evaluate_selectors(train, transfer or instance_test[:1], FULL),
        "ablation": ablation(train, instance_test),
        "importance": permutation_importance(train, instance_test),
    }
    return write_outputs(study, out_dir)

"""Log-runtime and log-memory regressors, plus a depth-limited selector tree."""
from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass


def _mae(y, pred) -> float:
    return sum(abs(a - b) for a, b in zip(y, pred)) / max(1, len(y))


def _rmse(y, pred) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(y, pred)) / max(1, len(y)))


def _r2(y, pred) -> float:
    if len(y) < 2:
        return 0.0
    mu = sum(y) / len(y)
    ss_tot = sum((v - mu) ** 2 for v in y)
    ss_res = sum((a - b) ** 2 for a, b in zip(y, pred))
    return 0.0 if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot


@dataclass
class Score:
    name: str
    target: str
    mae: float
    rmse: float
    r2: float


def score(name: str, target: str, y, pred) -> Score:
    return Score(name, target, _mae(y, pred), _rmse(y, pred), _r2(y, pred))


class _Scaled:
    def __init__(self) -> None:
        self.mean: list[float] = []
        self.std: list[float] = []

    def fit_transform(self, X: list[list[float]]) -> list[list[float]]:
        d = len(X[0])
        self.mean = [sum(row[j] for row in X) / len(X) for j in range(d)]
        self.std = []
        for j in range(d):
            var = sum((row[j] - self.mean[j]) ** 2 for row in X) / len(X)
            self.std.append(math.sqrt(var) or 1.0)
        return self.transform(X)

    def transform(self, X: list[list[float]]) -> list[list[float]]:
        return [[(row[j] - self.mean[j]) / self.std[j] for j in range(len(row))] for row in X]


class MedianBaseline:
    name = "median_baseline"

    def __init__(self) -> None:
        self.value = 1.0

    def fit(self, X, y):
        ordered = sorted(y)
        self.value = ordered[len(ordered) // 2] if ordered else 1.0
        return self

    def predict(self, X) -> list[float]:
        return [self.value] * len(X)


class LinearLog:
    """Least squares on log10(y + eps), predictions converted back to the original unit."""

    name = "linear_regression"

    def __init__(self, eps: float = 1e-3) -> None:
        self.eps = eps
        self.w = None
        self.scale = _Scaled()

    def fit(self, X, y):
        import numpy as np

        Z = self.scale.fit_transform(X)
        A = np.column_stack([np.ones(len(Z)), np.asarray(Z, dtype=float)])
        target = np.log10(np.asarray(y, dtype=float) + self.eps)
        self.w, *_ = np.linalg.lstsq(A, target, rcond=None)
        return self

    def predict(self, X) -> list[float]:
        import numpy as np

        Z = self.scale.transform(X)
        A = np.column_stack([np.ones(len(Z)), np.asarray(Z, dtype=float)])
        logged = A @ self.w
        # Clamp log-space predictions so size-shift extrapolation cannot overflow.
        return [max(0.0, float(10 ** float(min(6.0, max(-6.0, z)))) - self.eps) for z in logged]

    def coefficients(self) -> list[float]:
        return [float(v) for v in self.w[1:]]


def _variance(y) -> float:
    if len(y) < 2:
        return 0.0
    mu = sum(y) / len(y)
    return sum((v - mu) ** 2 for v in y) / len(y)


def _reg_tree(X, y, depth, max_depth, rng, min_leaf=4):
    if depth >= max_depth or len(y) < min_leaf * 2:
        return ("leaf", sum(y) / len(y) if y else 0.0)
    parent = _variance(y)
    d = len(X[0])
    feats = list(range(d))
    rng.shuffle(feats)
    best = None
    for j in feats[: max(1, int(math.sqrt(d)))]:
        vals = sorted({row[j] for row in X})
        if len(vals) < 2:
            continue
        mids = [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)]
        if len(mids) > 6:
            mids = mids[:: max(1, len(mids) // 6)]
        for t in mids:
            left = [i for i, row in enumerate(X) if row[j] <= t]
            right = [i for i in range(len(X)) if i not in set(left)]
            if len(left) < min_leaf or len(right) < min_leaf:
                continue
            var = (len(left) * _variance([y[i] for i in left]) + len(right) * _variance([y[i] for i in right])) / len(y)
            gain = parent - var
            if best is None or gain > best[0]:
                best = (gain, j, t, left, right)
    if best is None or best[0] <= 1e-8:
        return ("leaf", sum(y) / len(y))
    _, j, t, left, right = best
    return (
        "split",
        j,
        t,
        _reg_tree([X[i] for i in left], [y[i] for i in left], depth + 1, max_depth, rng, min_leaf),
        _reg_tree([X[i] for i in right], [y[i] for i in right], depth + 1, max_depth, rng, min_leaf),
    )


def _pred_reg(node, row) -> float:
    while node[0] == "split":
        _, j, t, left, right = node
        node = left if row[j] <= t else right
    return node[1]


class ForestLog:
    name = "random_forest"

    def __init__(self, trees: int = 8, depth: int = 4, seed: int = 0, eps: float = 1e-3) -> None:
        self.trees_n = trees
        self.depth = depth
        self.seed = seed
        self.eps = eps
        self.nodes = []

    def fit(self, X, y):
        rng = random.Random(self.seed)
        logged = [math.log10(v + self.eps) for v in y]
        self.nodes = []
        n = len(X)
        for _ in range(self.trees_n):
            idx = [rng.randrange(n) for _ in range(n)]
            self.nodes.append(
                _reg_tree([X[i] for i in idx], [logged[i] for i in idx], 0, self.depth, rng)
            )
        return self

    def predict(self, X) -> list[float]:
        out = []
        for row in X:
            z = sum(_pred_reg(node, row) for node in self.nodes) / len(self.nodes)
            out.append(max(0.0, 10 ** z - self.eps))
        return out


def _gini(labels: list[str]) -> float:
    counts = Counter(labels)
    n = len(labels)
    return 1.0 - sum((c / n) ** 2 for c in counts.values())


def _majority(labels: list[str]) -> str:
    return Counter(labels).most_common(1)[0][0]


def _cls_tree(X, y, depth, max_depth, rng):
    if depth >= max_depth or len(set(y)) == 1 or len(y) < 6:
        return ("leaf", _majority(y))
    parent = _gini(y)
    d = len(X[0])
    feats = list(range(d))
    rng.shuffle(feats)
    best = None
    for j in feats[: max(1, d)]:
        vals = sorted({row[j] for row in X})
        if len(vals) < 2:
            continue
        mids = [(vals[i] + vals[i + 1]) / 2 for i in range(0, len(vals) - 1, max(1, len(vals) // 5))]
        for t in mids:
            left = [i for i, row in enumerate(X) if row[j] <= t]
            right = [i for i in range(len(X)) if i not in set(left)]
            if len(left) < 2 or len(right) < 2:
                continue
            g = (len(left) * _gini([y[i] for i in left]) + len(right) * _gini([y[i] for i in right])) / len(y)
            gain = parent - g
            if best is None or gain > best[0]:
                best = (gain, j, t, left, right)
    if best is None or best[0] <= 1e-8:
        return ("leaf", _majority(y))
    _, j, t, left, right = best
    return (
        "split",
        j,
        t,
        _cls_tree([X[i] for i in left], [y[i] for i in left], depth + 1, max_depth, rng),
        _cls_tree([X[i] for i in right], [y[i] for i in right], depth + 1, max_depth, rng),
    )


def _pred_cls(node, row) -> str:
    while node[0] == "split":
        _, j, t, left, right = node
        node = left if row[j] <= t else right
    return node[1]


class TreeClassifier:
    name = "decision_tree"

    def __init__(self, depth: int = 4, seed: int = 0) -> None:
        self.depth = depth
        self.seed = seed
        self.node = ("leaf", "dijkstra_heap")

    def fit(self, X, y):
        self.node = _cls_tree(X, y, 0, self.depth, random.Random(self.seed))
        return self

    def predict(self, X) -> list[str]:
        return [_pred_cls(self.node, row) for row in X]


def loglog_slope(sizes: list[float], times: list[float]) -> float:
    pts = [(math.log(n), math.log(t)) for n, t in zip(sizes, times) if n > 1 and t > 0]
    if len(pts) < 2:
        return 0.0
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    num = sum((x - mx) * (y - my) for x, y in pts)
    den = sum((x - mx) ** 2 for x, _y in pts)
    return num / den if den else 0.0

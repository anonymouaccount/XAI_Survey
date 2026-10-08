

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, Optional, Sequence

import numpy as np

FeatureSet = FrozenSet[int]
ExplainFn = Callable[[Any, int], np.ndarray]


# ---------------------------------------------------------------------------
# Task: the modality-specific operations needed by the metrics
# ---------------------------------------------------------------------------

@dataclass
class Task:


    predict_proba: Callable[[Sequence[Any]], np.ndarray]
    remove: Callable[[Any, FeatureSet], Any]
    complete: Callable[[Any, FeatureSet, Any], Any]
    perturb: Callable[[Any, float, np.random.Generator], Any]
    background: Callable[[np.random.Generator, int], Sequence[Any]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_to_scores(indices, d: int) -> np.ndarray:
    v = np.zeros(d, dtype=float)
    idx = np.asarray(list(indices), dtype=int)
    if idx.size:
        v[idx] = 1.0
    return v


def top_k(scores: np.ndarray, k: int) -> FeatureSet:
   
    s = np.abs(np.asarray(scores, dtype=float).ravel())
    k = int(min(max(k, 0), s.size))
    order = np.argsort(-s, kind="stable")
    return frozenset(int(i) for i in order[:k])


def jaccard(a: FeatureSet, b: FeatureSet) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def predicted_class(task: Task, x: Any) -> tuple[int, float]:
    p = np.asarray(task.predict_proba([x]))[0]
    c = int(np.argmax(p))
    return c, float(p[c])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def fidelity_deletion(task: Task, x: Any, S: FeatureSet,
                      y_hat: Optional[int] = None) -> float:
    probs = np.asarray(task.predict_proba([x, task.remove(x, S)]))
    if y_hat is None:
        y_hat = int(np.argmax(probs[0]))
    p0, p1 = float(probs[0, y_hat]), float(probs[1, y_hat])
    if p0 <= 0.0:
        return 0.0
    return max(0.0, p0 - p1) / p0


def fidelity_sufficiency(task: Task, x: Any, S: FeatureSet,
                         backgrounds: Sequence[Any],
                         y_hat: Optional[int] = None) -> float:
    if y_hat is None:
        y_hat, _ = predicted_class(task, x)
    completed = [task.complete(x, S, z) for z in backgrounds]
    if not completed:
        return float("nan")
    preds = np.argmax(np.asarray(task.predict_proba(completed)), axis=1)
    return float(np.mean(preds == y_hat))


def gini_sparsity(scores: np.ndarray) -> float:
    a = np.sort(np.abs(np.asarray(scores, dtype=float).ravel()))
    n = a.size
    total = a.sum()
    if n == 0 or total <= 0:
        return 0.0
    j = np.arange(1, n + 1)
    return float(1.0 - 2.0 * np.sum((a / total) * (n - j + 0.5) / n))


def continuity(task: Task, explain: ExplainFn, x: Any, k: int, eps: float,
               rng: np.random.Generator, n: int = 20, max_tries: int = 100,
               seed: int = 0, S_ref: Optional[FeatureSet] = None) -> Dict[str, float]:
    y_hat, _ = predicted_class(task, x)
    if S_ref is None:
        S_ref = top_k(explain(x, seed), k)

    candidates = [task.perturb(x, eps, rng) for _ in range(max_tries)]
    preds = np.argmax(np.asarray(task.predict_proba(candidates)), axis=1)
    valid = [c for c, p in zip(candidates, preds) if p == y_hat][:n]

    overlaps = [jaccard(S_ref, top_k(explain(xp, seed), k)) for xp in valid]
    if not overlaps:
        return {"mean": float("nan"), "min": float("nan"), "n_valid": 0}
    return {"mean": float(np.mean(overlaps)), "min": float(np.min(overlaps)),
            "n_valid": len(overlaps)}


def consistency(explain: ExplainFn, x: Any, k: int,
                seeds: tuple[int, int] = (0, 1)) -> float:
    return jaccard(top_k(explain(x, seeds[0]), k), top_k(explain(x, seeds[1]), k))


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------

def make_random_explainer(n_features: Callable[[Any], int],
                          rng: np.random.Generator) -> ExplainFn:
    def explain(x: Any, seed: int = 0) -> np.ndarray:
        return rng.random(n_features(x))
    return explain


# ---------------------------------------------------------------------------
# All metrics for one instance
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    k: int = 5                 # explanation size (overridden per instance if needed)
    eps_stability: float = 0.05
    eps_robustness: float = 0.2
    n_perturbations: int = 20
    max_tries: int = 100
    n_background: int = 100    # M in Eq. (fid-suf)
    check_consistency: bool = True


def evaluate_instance(task: Task, explain: ExplainFn, x: Any, cfg: EvalConfig,
                      rng: np.random.Generator, seed: int = 0,
                      k: Optional[int] = None,
                      scores: Optional[np.ndarray] = None,
                      runtime: Optional[float] = None) -> Dict[str, float]:
    k = cfg.k if k is None else k
    y_hat, _ = predicted_class(task, x)

    if scores is None:
        t0 = time.perf_counter()
        scores = np.asarray(explain(x, seed), dtype=float).ravel()
        runtime = time.perf_counter() - t0
    scores = np.asarray(scores, dtype=float).ravel()
    S = top_k(scores, k)

    backgrounds = task.background(rng, cfg.n_background)
    stab = continuity(task, explain, x, k, cfg.eps_stability, rng,
                      cfg.n_perturbations, cfg.max_tries, seed, S_ref=S)
    rob = continuity(task, explain, x, k, cfg.eps_robustness, rng,
                     cfg.n_perturbations, cfg.max_tries, seed, S_ref=S)

    return {
        "k": len(S),
        "d": scores.size,
        "fid_del": fidelity_deletion(task, x, S, y_hat),
        "fid_suf": fidelity_sufficiency(task, x, S, backgrounds, y_hat),
        "sparsity": gini_sparsity(scores),
        "stability": stab["mean"],
        "robustness": rob["min"],
        "n_valid_stab": stab["n_valid"],
        "n_valid_rob": rob["n_valid"],
        "consistency": consistency(explain, x, k) if cfg.check_consistency else float("nan"),
        "time_s": float("nan") if runtime is None else runtime,
    }

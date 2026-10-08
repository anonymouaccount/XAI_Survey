
import numpy as np
import pytest

from xaibench.metrics import (EvalConfig, Task, consistency, continuity, evaluate_instance,
                              fidelity_deletion, fidelity_sufficiency, gini_sparsity,
                              jaccard, make_random_explainer, set_to_scores, top_k)

D = 10


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def toy_task():
    def predict_proba(xs):
        X = np.asarray(xs, dtype=float)
        p1 = sigmoid(5.0 * X[:, 0])
        return np.stack([1 - p1, p1], axis=1)

    def remove(x, S):
        x = np.array(x, dtype=float)
        x[list(S)] = 0.0
        return x

    def complete(x, S, z):
        out = np.array(z, dtype=float)
        idx = list(S)
        out[idx] = np.asarray(x)[idx]
        return out

    def perturb(x, eps, rng):
        return np.asarray(x) + rng.normal(0, eps, size=np.shape(x))

    def background(rng, m):
        return list(rng.normal(0, 1, size=(m, D)))

    return Task(predict_proba, remove, complete, perturb, background)


X0 = np.array([1.0] + [0.5] * (D - 1))    # predicted class 1 with high confidence


def good_explainer(x, seed=0):
    s = np.zeros(D)
    s[0] = 5.0 * x[0]
    return s


def bad_explainer(x, seed=0):
    s = np.zeros(D)
    s[1] = 1.0
    return s


# ---------------- helpers ----------------

def test_top_k_and_ties():
    assert top_k(np.array([0.1, -3.0, 2.0, 0.0]), 2) == frozenset({1, 2})
    assert top_k(np.zeros(4), 2) == frozenset({0, 1})           # ties -> lowest indices
    assert top_k(set_to_scores([3, 1], 5), 2) == frozenset({1, 3})


def test_jaccard():
    assert jaccard(frozenset({1, 2}), frozenset({2, 3})) == pytest.approx(1 / 3)
    assert jaccard(frozenset(), frozenset()) == 1.0


# ---------------- sparsity ----------------

def test_gini_uniform_is_zero():
    assert gini_sparsity(np.ones(D)) == pytest.approx(0.0)


def test_gini_indicator_equals_one_minus_m_over_d():
    for m in [1, 3, 7]:
        v = set_to_scores(range(m), D)
        assert gini_sparsity(v) == pytest.approx(1 - m / D)


def test_gini_scale_and_sign_invariant():
    v = np.random.default_rng(0).normal(size=D)
    assert gini_sparsity(v) == pytest.approx(gini_sparsity(-3.0 * v))


def test_gini_zero_vector():
    assert gini_sparsity(np.zeros(D)) == 0.0


# ---------------- fidelity ----------------

def test_deletion_fidelity_good_vs_bad():
    task = toy_task()
    good = fidelity_deletion(task, X0, frozenset({0}))
    bad = fidelity_deletion(task, X0, frozenset({1}))
    # removing x0 moves p from sigmoid(5)=0.993 to 0.5
    assert good == pytest.approx((sigmoid(5) - 0.5) / sigmoid(5), rel=1e-6)
    assert bad == pytest.approx(0.0)
    assert 0.0 <= bad <= good <= 1.0


def test_sufficiency_fidelity_good_vs_bad():
    task = toy_task()
    Z = task.background(np.random.default_rng(0), 500)
    assert fidelity_sufficiency(task, X0, frozenset({0}), Z) == pytest.approx(1.0)
    bad = fidelity_sufficiency(task, X0, frozenset({1}), Z)
    assert 0.35 < bad < 0.65          # the class is then decided by the random z0


# ---------------- stability, robustness, consistency ----------------

def test_deterministic_explainer_is_stable_and_consistent():
    task = toy_task()
    rng = np.random.default_rng(0)
    res = continuity(task, good_explainer, X0, k=1, eps=0.05, rng=rng)
    assert res["n_valid"] > 0
    assert res["mean"] == pytest.approx(1.0)
    assert res["min"] == pytest.approx(1.0)
    assert consistency(good_explainer, X0, k=1) == 1.0


def test_random_explainer_is_unstable_and_inconsistent():
    task = toy_task()
    rng = np.random.default_rng(0)
    rnd = make_random_explainer(lambda x: D, rng)
    res = continuity(task, rnd, X0, k=2, eps=0.05, rng=rng, n=50)
    assert res["mean"] < 0.4          # expected overlap of two random 2-sets out of 10 is ~0.13
    cons = np.mean([consistency(rnd, X0, k=2) for _ in range(50)])
    assert cons < 0.4


def test_only_prediction_preserving_perturbations_are_used():
    task = toy_task()
    x_border = np.array([0.01] + [0.0] * (D - 1))   # very close to the decision boundary
    res = continuity(task, good_explainer, x_border, k=1, eps=1.0,
                     rng=np.random.default_rng(0), n=20, max_tries=40)
    assert res["n_valid"] <= 20
    y = np.argmax(task.predict_proba([x_border])[0])
    assert y == 1


# ---------------- full evaluation ----------------

def test_evaluate_instance_ranges_and_ordering():
    task = toy_task()
    cfg = EvalConfig(k=1, n_background=200)
    good = evaluate_instance(task, good_explainer, X0, cfg, np.random.default_rng(0))
    bad = evaluate_instance(task, bad_explainer, X0, cfg, np.random.default_rng(0))
    for r in (good, bad):
        for key in ["fid_del", "fid_suf", "sparsity", "stability", "robustness", "consistency"]:
            assert 0.0 <= r[key] <= 1.0, key
    assert good["fid_del"] > bad["fid_del"]
    assert good["fid_suf"] > bad["fid_suf"]

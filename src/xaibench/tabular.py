

from __future__ import annotations

import json
import time
import warnings
import zipfile
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from xaibench.metrics import (EvalConfig, Task, evaluate_instance, jaccard,
                              make_random_explainer, set_to_scores, top_k)
from xaibench.utils import (DATA_DIR, RESULTS_DIR, ResultsWriter, get_logger,
                            run_metadata, set_seed)

warnings.filterwarnings("ignore")
log = get_logger("tabular")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TabularConfig:
    n_instances: int = 200          # explained test instances (balanced over predicted classes)
    test_size: float = 0.2
    # model
    n_estimators: int = 100
    max_depth: int = 4
    learning_rate: float = 0.1
    # metrics (Table 4 of the paper)
    eps_stability: float = 0.05     # Gaussian noise, in standard deviations of each numerical feature
    eps_robustness: float = 0.2
    n_perturbations: int = 10
    max_tries: int = 60
    n_background: int = 100
    k: int = 5                      # explanation size for attribution methods (Table 4)
    # explainers
    lime_samples: int = 1000
    anchor_threshold: float = 0.95
    pyxai_iterations: int = 50      # randomized runs of PyXAI's tree-specific algorithm

    @classmethod
    def smoke(cls) -> "TabularConfig":
        return cls(n_instances=4, n_estimators=30, max_depth=3, n_perturbations=3,
                   max_tries=15, n_background=30, lime_samples=300, pyxai_iterations=5)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

NUMERICAL = ["age", "time_in_hospital", "num_lab_procedures", "num_procedures",
             "num_medications", "number_outpatient", "number_emergency",
             "number_inpatient", "number_diagnoses"]
DRUGS = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
         "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
         "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "examide",
         "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
         "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]
OTHER_CATEGORICAL = ["race", "gender", "admission_type_id", "discharge_disposition_id",
                     "admission_source_id", "medical_specialty", "diag_1", "diag_2", "diag_3",
                     "max_glu_serum", "A1Cresult", "change", "diabetesMed"]
# Discharged to hospice or deceased: these patients cannot be readmitted (Strack et al., 2014)
EXCLUDED_DISCHARGE = {11, 13, 14, 19, 20, 21}


def icd9_group(code) -> str:
    if pd.isna(code) or code == "":
        return "Missing"
    if code[0] in "VE":
        return "Other"
    v = float(code)
    i = int(v)
    if 250 <= v < 251:
        return "Diabetes"
    if 390 <= i <= 459 or i == 785:
        return "Circulatory"
    if 460 <= i <= 519 or i == 786:
        return "Respiratory"
    if 520 <= i <= 579 or i == 787:
        return "Digestive"
    if 800 <= i <= 999:
        return "Injury"
    if 710 <= i <= 739:
        return "Musculoskeletal"
    if 580 <= i <= 629 or i == 788:
        return "Genitourinary"
    if 140 <= i <= 239:
        return "Neoplasms"
    return "Other"


def load_diabetes():
    with zipfile.ZipFile(path) as z:
        with z.open("diabetic_data.csv") as f:
            df = pd.read_csv(f, keep_default_na=False, na_values=["?"], low_memory=False)
    n_raw = len(df)

    # one encounter per patient (the first), so that a patient is never in both train and test
    df = df.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first")
    df = df[~df["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE)].copy()

    y = (df["readmitted"] != "NO").astype(int).to_numpy()

    df["age"] = df["age"].str.extract(r"\[(\d+)-").astype(float)[0] + 5   # "[70-80)" -> 75
    for c in ["diag_1", "diag_2", "diag_3"]:
        df[c] = df[c].map(icd9_group)
    top = df["medical_specialty"].value_counts().index[:10]
    df["medical_specialty"] = np.where(df["medical_specialty"].isna(), "Missing",
                                       np.where(df["medical_specialty"].isin(top),
                                                df["medical_specialty"], "Other"))
    # keep drugs prescribed (value != "No") to at least 1% of patients
    drugs = [d for d in DRUGS if (df[d] != "No").mean() >= 0.01]

    categorical = OTHER_CATEGORICAL + drugs
    columns = NUMERICAL + categorical
    X = pd.DataFrame(index=df.index)
    for c in NUMERICAL:
        X[c] = df[c].astype(float)
    cat_names = {}
    for c in categorical:
        values = df[c].astype(str).fillna("Missing")
        cats = sorted(values.unique())
        cat_names[columns.index(c)] = cats
        X[c] = pd.Categorical(values, categories=cats).codes.astype(float)
    X = X[columns]
    cat_idx = [columns.index(c) for c in categorical]
    log.info("Diabetes: %d raw encounters -> %d patients, %d features (%d numerical), "
             "positive rate %.3f", n_raw, len(X), X.shape[1], len(NUMERICAL), y.mean())
    return X, y, cat_idx, cat_names


# ---------------------------------------------------------------------------
# Task (modality-specific operations used by the metrics)
# ---------------------------------------------------------------------------

def make_task(model, X_train: np.ndarray, cat_idx, rng_background: np.random.Generator) -> Task:
    d = X_train.shape[1]
    num_idx = np.array([i for i in range(d) if i not in set(cat_idx)])
    baseline = np.median(X_train, axis=0)
    for i in cat_idx:                                   # mode for categorical features
        vals, counts = np.unique(X_train[:, i], return_counts=True)
        baseline[i] = vals[np.argmax(counts)]
    std = X_train.std(axis=0)
    lo, hi = X_train.min(axis=0), X_train.max(axis=0)

    def predict_proba(xs):
        return model.predict_proba(np.asarray(xs, dtype=float))

    def remove(x, S):
        out = np.array(x, dtype=float)
        idx = list(S)
        out[idx] = baseline[idx]
        return out

    def complete(x, S, z):
        out = np.array(z, dtype=float)
        idx = list(S)
        out[idx] = np.asarray(x)[idx]
        return out

    def perturb(x, eps, rng):
        out = np.array(x, dtype=float)
        out[num_idx] += rng.normal(0.0, eps * std[num_idx])
        out[num_idx] = np.clip(out[num_idx], lo[num_idx], hi[num_idx])
        return out

    def background(rng, m):
        rows = rng_background.choice(len(X_train), size=m, replace=False)
        return list(X_train[rows])

    return Task(predict_proba, remove, complete, perturb, background)


# ---------------------------------------------------------------------------
# Explainers: every explainer is explain(x, seed) -> score vector of length d
# ---------------------------------------------------------------------------

class PyXAIExplainer:

    def __init__(self, model, n_iterations: int):
        import sys
        argv, sys.argv = sys.argv, sys.argv[:1]     # PyXAI parses the command line when imported
        from pyxai import Explaining, Learning, Tools
        sys.argv = argv
        Tools.set_verbose(0)
        # PyXAI writes the model to a fixed file name ("xgboost_JSON.json") in the current
        # folder; parallel jobs would overwrite each other's file, so use a private folder.
        import os
        import tempfile
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                _, self.model = Learning.ModelIO.import_models(model, instances_type="tabular")
            finally:
                os.chdir(cwd)
        self._Explaining = Explaining
        self._explainer = None
        self.n_iterations = n_iterations
        # literal id -> 0-based feature index (PyXAI numbers features from 1)
        self._lit2feat = {i: m[0] - 1 for i, m in
                          enumerate(self.model.map_id_binaries_to_features) if i > 0}

    def _set(self, x):
        x = [float(v) for v in x]
        if self._explainer is None:
            self._explainer = self._Explaining.initialize(self.model, x)
        else:
            self._explainer.set_instance(x)
        return self._explainer

    def feature_set(self, x, seed: int = 0) -> frozenset:
        e = self._set(x)
        # PyXAI uses seed=0 for "random seed", so shift by one to stay deterministic
        reason = e.tree_specific_reason(n_iterations=self.n_iterations, seed=seed + 1, history=False)
        return frozenset(self._lit2feat[abs(l)] for l in reason)

    def explain(self, x, seed: int = 0) -> np.ndarray:
        return set_to_scores(self.feature_set(x, seed), len(x))

    def is_sufficient(self, x, S) -> bool:
        e = self._set(x)
        partial = tuple(l for l in e.binary_representation if self._lit2feat[abs(l)] in S)
        return bool(e.is_implicant(partial))


def make_treeshap(model):
    booster = model.get_booster()

    def explain(x, seed: int = 0):
        contrib = booster.predict(xgb.DMatrix(np.asarray(x, dtype=float)[None, :]),
                                  pred_contribs=True)
        return contrib[0, :-1]          # last column is the bias term
    return explain


def make_lime(model, X_train, feature_names, cat_idx, cat_names, n_samples):
    from lime.lime_tabular import LimeTabularExplainer
    lime = LimeTabularExplainer(X_train, feature_names=feature_names, class_names=["0", "1"],
                                categorical_features=cat_idx, categorical_names=cat_names,
                                discretize_continuous=True, mode="classification",
                                random_state=0)
    d = X_train.shape[1]

    def explain(x, seed: int = 0):
        lime.random_state = np.random.RandomState(seed)
        y_hat = int(model.predict(np.asarray(x)[None, :])[0])
        exp = lime.explain_instance(np.asarray(x), model.predict_proba, labels=(y_hat,),
                                    num_features=d, num_samples=n_samples)
        scores = np.zeros(d)
        for i, w in exp.as_map()[y_hat]:
            scores[i] = w
        return scores
    return explain


def make_anchors(model, X_train, feature_names, cat_names, threshold):
    from anchor import anchor_tabular
    anchors = anchor_tabular.AnchorTabularExplainer(["0", "1"], feature_names, X_train, cat_names)
    d = X_train.shape[1]
    last = {}

    def predict(a):
        return model.predict(np.asarray(a, dtype=float))

    def explain(x, seed: int = 0):
        np.random.seed(seed)                      # Anchors uses NumPy's global generator
        exp = anchors.explain_instance(np.asarray(x), predict, threshold=threshold)
        last["precision"] = float(exp.precision())
        last["rule"] = " AND ".join(exp.names())      # human-readable rule (for figures)
        return set_to_scores(exp.features(), d)
    explain.last = last
    return explain


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def smallest_sufficient_k(pyxai: "PyXAIExplainer", x, scores) -> int:
    lo, hi = 1, len(scores)                 # all features together are always sufficient
    while lo < hi:
        mid = (lo + hi) // 2
        if pyxai.is_sufficient(x, top_k(scores, mid)):
            hi = mid
        else:
            lo = mid + 1
    return lo


def select_instances(model, X_test, y_test, n, rng):
    pred = model.predict(X_test)
    correct = np.flatnonzero(pred == y_test)
    chosen = []
    for c in (0, 1):
        pool = correct[pred[correct] == c]
        chosen += list(rng.choice(pool, size=min(n // 2, len(pool)), replace=False))
    return np.array(sorted(chosen))


def main(seed: int = 0, smoke: bool = False) -> None:
    cfg = TabularConfig.smoke() if smoke else TabularConfig()
    rng = set_seed(seed)
    out_dir = RESULTS_DIR / "tabular" / ("smoke" if smoke else "")
    writer = ResultsWriter(out_dir / f"seed{seed}.csv")
    meta = run_metadata("tabular", seed)
    log.info("Config: %s", asdict(cfg))

    # ---- data and ONE model --------------------------------------------------
    X_df, y, cat_idx, cat_names = load_diabetes()
    feature_names = list(X_df.columns)
    X = X_df.to_numpy(dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=seed)
    model = xgb.XGBClassifier(n_estimators=cfg.n_estimators, max_depth=cfg.max_depth,
                              learning_rate=cfg.learning_rate, subsample=0.8, scale_pos_weight=float((y_train == 0).sum() / (y_train == 1).sum()),
                              random_state=seed, n_jobs=-1, eval_metric="logloss")
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    perf = {"accuracy": accuracy_score(y_test, proba > 0.5),
            "balanced_accuracy": balanced_accuracy_score(y_test, proba > 0.5),
            "auc": roc_auc_score(y_test, proba),
            "n_train": len(y_train), "n_test": len(y_test), "n_features": X.shape[1],
            "positive_rate": float(y.mean()), **meta, "config": asdict(cfg)}
    (out_dir).mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"model_seed{seed}.json", "w") as f:
        json.dump(perf, f, indent=2, default=str)
    log.info("Model: accuracy %.3f | balanced accuracy %.3f | AUC %.3f",
             perf["accuracy"], perf["balanced_accuracy"], perf["auc"])

    # ---- task and explainers ---------------------------------------------------
    task = make_task(model, X_train, cat_idx, np.random.default_rng(seed + 1000))
    ecfg = EvalConfig(eps_stability=cfg.eps_stability, eps_robustness=cfg.eps_robustness,
                      n_perturbations=cfg.n_perturbations, max_tries=cfg.max_tries,
                      n_background=cfg.n_background)
    pyxai = PyXAIExplainer(model, cfg.pyxai_iterations)
    anchors = make_anchors(model, X_train, feature_names, cat_names, cfg.anchor_threshold)
    methods = {
        "pyxai": (pyxai.explain, "set"),
        "anchors": (anchors, "set"),
        "lime": (make_lime(model, X_train, feature_names, cat_idx, cat_names, cfg.lime_samples), "attribution"),
        "treeshap": (make_treeshap(model), "attribution"),
        "random": (make_random_explainer(lambda x: X.shape[1], np.random.default_rng(seed + 2000)), "attribution"),
    }

    # ---- explain the same instances with every method ---------------------------
    idx = select_instances(model, X_test, y_test, cfg.n_instances, rng)
    log.info("Explaining %d instances with %s", len(idx), list(methods))
    for n_done, i in enumerate(idx, start=1):
        x = X_test[i]
        p = model.predict_proba(x[None, :])[0]
        t0 = time.perf_counter()
        E = pyxai.feature_set(x, seed)
        t_pyxai = time.perf_counter() - t0
        k_x = len(E)

        for name, (explain, kind) in methods.items():
            t0 = time.perf_counter()
            scores = np.asarray(explain(x, seed), dtype=float)
            runtime = t_pyxai if name == "pyxai" else time.perf_counter() - t0
            precision = anchors.last.get("precision") if name == "anchors" else np.nan
            # main metrics: attribution methods at k = cfg.k, set-valued methods on their own set
            k = int(round(scores.sum())) if kind == "set" else cfg.k
            S = top_k(scores, k)
            res = evaluate_instance(task, explain, x, ecfg, rng, seed=seed, k=k,
                                    scores=scores, runtime=runtime)
            # formal comparison with the abductive explanation E(x)
            if kind == "attribution":
                S_E = top_k(scores, k_x)                       # same size as E(x)
                suff_kE = pyxai.is_sufficient(x, S_E)
                k_min = smallest_sufficient_k(pyxai, x, scores)
                jac = jaccard(S_E, E)
            else:
                suff_kE, k_min, jac = np.nan, (k_x if name == "pyxai" else np.nan), jaccard(S, E)
            row = {**meta, "instance": int(i), "y_true": int(y_test[i]),
                   "y_pred": int(np.argmax(p)), "p_pred": float(p.max()),
                   "method": name, "kind": kind, **res,
                   "size_abductive": k_x,
                   "formally_sufficient": pyxai.is_sufficient(x, S),
                   "sufficient_at_size_E": suff_kE,
                   "k_min_sufficient": k_min,
                   "jaccard_abductive": jac,
                   "anchor_precision": precision}
            writer.write(row)
        log.info("[%d/%d] instance %d done (|E(x)| = %d features)", n_done, len(idx), i, k_x)

    log.info("Results written to %s", writer.path)

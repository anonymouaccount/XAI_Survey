

from __future__ import annotations

import copy
import json
import time
import warnings
import zipfile
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from xaibench.metrics import (EvalConfig, Task, evaluate_instance,
                              make_random_explainer, top_k)
from xaibench.utils import (DATA_DIR, RESULTS_DIR, ResultsWriter, get_logger,
                            run_metadata, set_seed)

warnings.filterwarnings("ignore")
log = get_logger("timeseries")

# Measured variables used as inputs. Lat/Lon are constant, Station is a name,
# Cum_Rain is a running sum of Rain (redundant with past Rain).
MEASURED = ["MaxT", "MinT", "RH1", "RH2", "Wind", "Rain", "SSH", "Evap",
            "Radiation", "FAO56_ET"]
# The day of the year is NOT given as an input: it is the same for the 14 days
# of a window, so it filled most of the top-k entries of every explainer
# (smoke run) and hid the weather variables. The season is still visible to
# the model through the variables themselves (temperature, humidity, ...).
VARIABLES = MEASURED


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TimeSeriesConfig:
    n_instances: int = 200          # explained test windows (balanced over predicted classes)
    window: int = 14                # W days of history
    rain_threshold: float = 0.0     # rain day: Rain > threshold (mm)
    train_frac: float = 0.7
    val_frac: float = 0.1
    # model
    hidden: int = 64
    dropout: float = 0.2
    epochs: int = 40
    patience: int = 6
    batch_size: int = 256
    lr: float = 1e-3
    # metrics (Table 4 of the paper)
    k_frac: float = 0.10            # k = 10% of the W x V entries
    eps_stability: float = 0.05     # Gaussian jitter, in standard deviations
    eps_robustness: float = 0.2
    n_perturbations: int = 10
    max_tries: int = 60
    n_background: int = 100
    # explainers
    ig_steps: int = 50
    gshap_samples: int = 50
    gshap_baselines: int = 100
    lime_samples: int = 1000
    lime_segment: int = 2           # days per LIME time segment

    @classmethod
    def smoke(cls) -> "TimeSeriesConfig":
        return cls(n_instances=4, epochs=3, n_perturbations=3, max_tries=20,
                   n_background=20, ig_steps=10, gshap_samples=10,
                   gshap_baselines=20, lime_samples=100)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_weather() -> pd.DataFrame:
    """Daily records sorted by date, with no missing values."""
    path = DATA_DIR / "raw" / "daily-weather-data-40-years.zip"
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.lower().endswith((".xlsx", ".xls")))
        with z.open(name) as f:
            df = pd.read_excel(f)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    n_missing = int(df[MEASURED].isna().sum().sum())
    df[MEASURED] = df[MEASURED].astype(float).interpolate(limit_direction="both")
    log.info("Weather data: %d days from %s to %s, %d missing value(s) interpolated",
             len(df), df["Date"].iloc[0].date(), df["Date"].iloc[-1].date(), n_missing)
    return df


def make_windows(df: pd.DataFrame, W: int, threshold: float):
  
    values = df[VARIABLES].to_numpy(dtype=np.float32)
    rain = df["Rain"].to_numpy()
    day = (df["Date"] - df["Date"].iloc[0]).dt.days.to_numpy()
    X, y, dates = [], [], []
    for t in range(W - 1, len(df) - 1):
        if day[t + 1] - day[t - W + 1] != W:      # a gap inside the window
            continue
        X.append(values[t - W + 1:t + 1])
        y.append(int(rain[t + 1] > threshold))
        dates.append(df["Date"].iloc[t + 1])
    return np.stack(X), np.array(y), np.array(dates)


def chronological_split(n: int, train_frac: float, val_frac: float):
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return (np.arange(0, n_train), np.arange(n_train, n_train + n_val),
            np.arange(n_train + n_val, n))


def standardise(X: np.ndarray, train_idx: np.ndarray):
   
    flat = X[train_idx].reshape(-1, X.shape[2])
    mean, std = flat.mean(0), flat.std(0)
    std[std == 0] = 1.0
    return ((X - mean) / std).astype(np.float32), mean, std


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(n_vars: int, hidden: int, dropout: float):
    import torch.nn as nn

    class RainLSTM(nn.Module):

        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_vars, hidden, batch_first=True)
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(hidden, 2)

        def forward(self, x):                       # x: (batch, W, V)
            out, _ = self.lstm(x)
            return self.head(self.dropout(out[:, -1]))

    return RainLSTM()


def predict_proba_array(model, X: np.ndarray, batch: int = 4096) -> np.ndarray:
    import torch
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            logits = model(torch.as_tensor(X[i:i + batch], dtype=torch.float32))
            probs.append(torch.softmax(logits, dim=1).numpy())
    return np.concatenate(probs) if probs else np.zeros((0, 2))


def train_model(model, X, y, train_idx, val_idx, cfg: TimeSeriesConfig, seed: int):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    counts = np.bincount(y[train_idx], minlength=2)
    weights = torch.tensor(len(train_idx) / (2.0 * np.maximum(counts, 1)), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    Xt = torch.as_tensor(X[train_idx]); yt = torch.as_tensor(y[train_idx])
    gen = torch.Generator().manual_seed(seed)
    best_auc, best_state, bad = -1.0, None, 0
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=gen)
        for i in range(0, len(perm), cfg.batch_size):
            b = perm[i:i + cfg.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()
        val_auc = roc_auc_score(y[val_idx], predict_proba_array(model, X[val_idx])[:, 1])
        log.info("epoch %2d  loss %.4f  validation AUC %.4f", epoch + 1, loss.item(), val_auc)
        if val_auc > best_auc:
            best_auc, best_state, bad = val_auc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                log.info("early stopping at epoch %d", epoch + 1)
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best_auc


# ---------------------------------------------------------------------------
# Task: operations used by the metrics (features = flattened (day, variable) entries)
# ---------------------------------------------------------------------------

def make_task(model, X_train: np.ndarray, rng_background: np.random.Generator) -> Task:
    W, V = X_train.shape[1:]

    def predict_proba(xs):
        return predict_proba_array(model, np.stack([np.asarray(x, dtype=np.float32) for x in xs]))

    def remove(x, S):
        z = x.copy().reshape(-1)
        z[list(S)] = 0.0                              # training mean of the variable
        return z.reshape(W, V)

    def complete(x, S, z):
        out = np.asarray(z, dtype=np.float32).copy().reshape(-1)
        idx = list(S)
        out[idx] = x.reshape(-1)[idx]
        return out.reshape(W, V)

    def perturb(x, eps, rng):
        z = x.copy()
        z += rng.normal(0.0, eps, size=(W, V)).astype(np.float32)
        return z

    def background(rng, m):
        return list(X_train[rng_background.choice(len(X_train), size=m, replace=False)])

    return Task(predict_proba, remove, complete, perturb, background)


# ---------------------------------------------------------------------------
# Explainers: explain(x, seed) -> scores of length W*V (higher = more important)
# Scores are the attributions to the probability of the predicted class.
# ---------------------------------------------------------------------------

def _prob_fn(model):
    import torch

    def f(inp):
        return torch.softmax(model(inp), dim=1)
    return f


def make_integrated_gradients(model, n_steps: int):
    import torch
    from captum.attr import IntegratedGradients
    ig = IntegratedGradients(_prob_fn(model))

    def explain(x, seed: int = 0):
        inp = torch.as_tensor(x[None], dtype=torch.float32)
        target = int(predict_proba_array(model, x[None]).argmax())
        attr = ig.attribute(inp, baselines=torch.zeros_like(inp), target=target, n_steps=n_steps)
        return attr.detach().numpy().reshape(-1)
    return explain


def make_gradient_shap(model, X_train: np.ndarray, n_samples: int, n_baselines: int, seed: int):
    import torch
    from captum.attr import GradientShap
    gs = GradientShap(_prob_fn(model))
    rng = np.random.default_rng(seed + 3000)
    baselines = torch.as_tensor(X_train[rng.choice(len(X_train), n_baselines, replace=False)])

    def explain(x, seed: int = 0):
        torch.manual_seed(seed)
        inp = torch.as_tensor(x[None], dtype=torch.float32)
        target = int(predict_proba_array(model, x[None]).argmax())
        attr = gs.attribute(inp, baselines=baselines, n_samples=n_samples,
                            stdevs=0.0, target=target)
        return attr.detach().numpy().reshape(-1)
    return explain


def lime_blocks(W: int, V: int, segment: int) -> np.ndarray:
    n_seg = int(np.ceil(W / segment))
    block = np.empty((W, V), dtype=int)
    for t in range(W):
        for v in range(V):
            block[t, v] = v * n_seg + t // segment
    return block.reshape(-1)


def make_lime(model, W: int, V: int, segment: int, n_samples: int):
    from lime.lime_base import LimeBase
    from sklearn.metrics import pairwise_distances
    block = lime_blocks(W, V, segment)
    m = block.max() + 1

    def kernel(d, width=0.25):
        return np.sqrt(np.exp(-(d ** 2) / width ** 2))

    def explain(x, seed: int = 0):
        rng = np.random.default_rng(seed)
        masks = rng.integers(0, 2, size=(n_samples, m))
        masks[0, :] = 1                                          # the instance itself
        flat = x.reshape(-1)
        inputs = np.repeat(flat[None, :], n_samples, axis=0) * masks[:, block]
        probs = predict_proba_array(model, inputs.reshape(n_samples, W, V).astype(np.float32))
        target = int(probs[0].argmax())
        distances = pairwise_distances(masks, masks[:1], metric="cosine").ravel()
        base = LimeBase(kernel, random_state=seed)
        _, weights, _, _ = base.explain_instance_with_data(
            masks, probs, distances, target, num_features=m, feature_selection="none")
        block_weight = np.zeros(m)
        for j, w in weights:
            block_weight[j] = w
        return block_weight[block]                              # every entry gets its block weight
    return explain


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def select_instances(proba: np.ndarray, y: np.ndarray, n: int, rng):
    pred = proba.argmax(1)
    correct = np.flatnonzero(pred == y)
    chosen = []
    for c in (0, 1):
        pool = correct[pred[correct] == c]
        chosen += list(rng.choice(pool, size=min(n // 2, len(pool)), replace=False))
    return np.array(sorted(chosen))


def describe_top_k(S, W: int, V: int):
    S = np.array(sorted(S))
    lags = (W - 1) - S // V                     # 0 = day t, W-1 = oldest day
    variables = S % V
    top_var = VARIABLES[np.bincount(variables, minlength=V).argmax()]
    return float(lags.mean()), top_var


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(seed: int = 0, smoke: bool = False) -> None:
    import torch
    cfg = TimeSeriesConfig.smoke() if smoke else TimeSeriesConfig()
    rng = set_seed(seed)
    torch.manual_seed(seed)
    out_dir = RESULTS_DIR / "timeseries" / ("smoke" if smoke else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = ResultsWriter(out_dir / f"seed{seed}.csv")
    meta = run_metadata("timeseries", seed)
    log.info("Config: %s", asdict(cfg))

    # ---- data and ONE model ------------------------------------------------------
    df = load_weather()
    X_raw, y, dates = make_windows(df, cfg.window, cfg.rain_threshold)
    train_idx, val_idx, test_idx = chronological_split(len(y), cfg.train_frac, cfg.val_frac)
    X, mean, std = standardise(X_raw, train_idx)
    W, V = X.shape[1:]
    log.info("Windows: %d (train %d, validation %d, test %d), shape %d days x %d variables, "
             "rain days: %.1f%%", len(y), len(train_idx), len(val_idx), len(test_idx),
             W, V, 100 * y.mean())

    model = build_model(V, cfg.hidden, cfg.dropout)
    model, val_auc = train_model(model, X, y, train_idx, val_idx, cfg, seed)
    proba = predict_proba_array(model, X[test_idx])
    y_test = y[test_idx]
    perf = {"accuracy": accuracy_score(y_test, proba.argmax(1)),
            "balanced_accuracy": balanced_accuracy_score(y_test, proba.argmax(1)),
            "auc": roc_auc_score(y_test, proba[:, 1]),
            "validation_auc": val_auc,
            "majority_class_accuracy": float(max(y_test.mean(), 1 - y_test.mean())),
            "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
            "train_period": f"{dates[train_idx[0]].date()} to {dates[train_idx[-1]].date()}",
            "val_period": f"{dates[val_idx[0]].date()} to {dates[val_idx[-1]].date()}",
            "test_period": f"{dates[test_idx[0]].date()} to {dates[test_idx[-1]].date()}",
            "window_days": W, "n_variables": V, "variables": VARIABLES,
            "positive_rate_train": float(y[train_idx].mean()),
            "positive_rate_test": float(y_test.mean()),
            **meta, "config": asdict(cfg)}
    with open(out_dir / f"model_seed{seed}.json", "w") as f:
        json.dump(perf, f, indent=2, default=str)
    log.info("Model (test %s): accuracy %.3f | balanced accuracy %.3f | AUC %.3f",
             perf["test_period"], perf["accuracy"], perf["balanced_accuracy"], perf["auc"])

    # ---- task and explainers -------------------------------------------------------
    X_train = X[train_idx]
    task = make_task(model, X_train, np.random.default_rng(seed + 1000))
    k = max(1, int(round(cfg.k_frac * W * V)))
    ecfg = EvalConfig(k=k, eps_stability=cfg.eps_stability, eps_robustness=cfg.eps_robustness,
                      n_perturbations=cfg.n_perturbations, max_tries=cfg.max_tries,
                      n_background=cfg.n_background)
    methods = {
        "integrated_gradients": make_integrated_gradients(model, cfg.ig_steps),
        "gradient_shap": make_gradient_shap(model, X_train, cfg.gshap_samples, cfg.gshap_baselines, seed),
        "lime": make_lime(model, W, V, cfg.lime_segment, cfg.lime_samples),
        "random": make_random_explainer(lambda x: W * V, np.random.default_rng(seed + 2000)),
    }
    maps = {name: np.zeros((W, V)) for name in methods}
    example = {}

    # ---- explain the same windows with every method ----------------------------------
    chosen = select_instances(proba, y_test, cfg.n_instances, rng)
    log.info("Explaining %d test windows with %s (k = %d of %d entries)",
             len(chosen), list(methods), k, W * V)
    for n_done, j in enumerate(chosen, start=1):
        x = X[test_idx[j]]
        p = proba[j]
        for name, explain in methods.items():
            t0 = time.perf_counter()
            scores = np.asarray(explain(x, seed), dtype=float)
            runtime = time.perf_counter() - t0
            res = evaluate_instance(task, explain, x, ecfg, rng, seed=seed, k=k,
                                    scores=scores, runtime=runtime)
            mean_lag, top_var = describe_top_k(top_k(scores, k), W, V)
            maps[name] += np.abs(scores).reshape(W, V)
            if n_done == 1:
                example[name] = scores.reshape(W, V)
            writer.write({**meta, "instance": int(test_idx[j]),
                          "target_date": str(dates[test_idx[j]].date()),
                          "y_true": int(y_test[j]), "y_pred": int(p.argmax()),
                          "p_pred": float(p.max()), "method": name, "kind": "attribution",
                          **res, "mean_lag_topk": mean_lag, "top_variable": top_var})
        log.info("[%d/%d] window ending %s done", n_done, len(chosen),
                 dates[test_idx[j]].date())

    np.savez(out_dir / f"maps_seed{seed}.npz", variables=np.array(VARIABLES),
             example_input=X[test_idx[chosen[0]]] * std + mean,
             **{f"mean_abs_{n}": v / len(chosen) for n, v in maps.items()},
             **{f"example_{n}": v for n, v in example.items()})
    log.info("Results written to %s", writer.path)


import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import xgboost as xgb  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from xaibench import tabular as T  # noqa: E402
from xaibench.metrics import top_k  # noqa: E402
from xaibench.utils import RESULTS_DIR, set_seed  # noqa: E402


def value_label(name, v, i, cat_names):
    if i in cat_names:
        text = str(cat_names[i][int(v)])
    elif name == "age":
        text = f"{int(v) - 5}-{int(v) + 5}"
    else:
        text = f"{v:g}"
    text = text if len(text) <= 16 else text[:15] + "."
    return f"{name} = {text}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=int, default=5744,
                        help="test index of the patient (column 'instance' of seed0.csv)")
    args = parser.parse_args()
    seed = 0
    cfg = T.TabularConfig()
    set_seed(seed)

    # ---- the SAME data split and model as in tabular.main (seed 0) ---------------
    X_df, y, cat_idx, cat_names = T.load_diabetes()
    names = list(X_df.columns)
    X = X_df.to_numpy(dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=seed)
    model = xgb.XGBClassifier(n_estimators=cfg.n_estimators, max_depth=cfg.max_depth,
                              learning_rate=cfg.learning_rate, subsample=0.8,
                              scale_pos_weight=float((y_train == 0).sum() / (y_train == 1).sum()),
                              random_state=seed, n_jobs=-1, eval_metric="logloss")
    model.fit(X_train, y_train)

    x = X_test[args.instance]
    p = model.predict_proba(x[None, :])[0]
    y_hat = int(p.argmax())
    csv = RESULTS_DIR / "tabular" / "seed0.csv"
    if csv.exists():
        rows = pd.read_csv(csv)
        rows = rows[rows["instance"] == args.instance]
        if len(rows):
            print(f"Check: probability now {p.max():.4f}, in seed0.csv {rows['p_pred'].iloc[0]:.4f} "
                  "(must be equal: same model)")
        else:
            print(f"Note: patient {args.instance} is not among the explained patients of seed0.csv")

    # ---- explanations ------------------------------------------------------------
    pyxai = T.PyXAIExplainer(model, cfg.pyxai_iterations)
    E = pyxai.feature_set(x, seed)
    anchors = T.make_anchors(model, X_train, names, cat_names, cfg.anchor_threshold)
    A = set(np.flatnonzero(anchors(x, seed)))
    lime = T.make_lime(model, X_train, names, cat_idx, cat_names, cfg.lime_samples)(x, seed)
    shap = T.make_treeshap(model)(x, seed)
    shap_pred = shap if y_hat == 1 else -shap          # > 0 = towards the predicted class
    k_min = T.smallest_sufficient_k(pyxai, x, shap)

    columns = [
        (f"PyXAI\n|E(x)| = {len(E)}", set(E), None),
        (f"Anchors\n{len(A)} features", A, None),
        ("LIME\ntop 5", set(top_k(lime, 5)), lime),
        ("TreeSHAP\ntop 5", set(top_k(shap, 5)), shap_pred),
        (f"TreeSHAP\ntop {k_min}", set(top_k(shap, k_min)), shap_pred),
    ]
    sufficient = [pyxai.is_sufficient(x, S) for _, S, _ in columns]

    # ---- figure --------------------------------------------------------------------
    order = np.argsort(-np.abs(shap))                       # rows sorted by |TreeSHAP|
    d = len(names)
    grid = np.full((d, len(columns)), np.nan)
    for c, (_, S, scores) in enumerate(columns):
        for r, i in enumerate(order):
            if i in S:
                if scores is None:
                    grid[r, c] = 0.0                        # selected, no sign
                else:
                    m = np.abs(scores).max()
                    grid[r, c] = scores[i] / m if m > 0 else 0.0

    fig, ax = plt.subplots(figsize=(6.4, 7.6), constrained_layout=True)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("white")
    for c in range(len(columns)):
        for r in range(d):
            v = grid[r, c]
            if np.isnan(v):
                continue
            colour = "0.55" if columns[c][2] is None else cmap(0.5 + 0.5 * v)
            ax.add_patch(plt.Rectangle((c - 0.45, r - 0.42), 0.9, 0.84, facecolor=colour,
                                       edgecolor="0.45", linewidth=0.5))
    ax.set_xlim(-0.5, len(columns) - 0.5)
    ax.set_ylim(d - 0.5, -2.2)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([c[0] for c in columns], fontsize=7)
    ax.xaxis.tick_top()
    ax.set_yticks(range(d))
    ax.set_yticklabels([value_label(names[i], x[i], i, cat_names) for i in order], fontsize=6.5)
    for c, ok in enumerate(sufficient):
        ax.text(c, -1.3, "sufficient: yes" if ok else "sufficient: no", ha="center", va="center",
                fontsize=7, fontweight="bold", color="darkgreen" if ok else "darkred")
    ax.axhline(-0.6, color="0.3", lw=0.6)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f"Patient {args.instance}: predicted "
                 f"{'readmitted' if y_hat else 'not readmitted'} (p = {p.max():.2f})",
                 fontsize=8, pad=4)

    out = RESULTS_DIR / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"tabular_patient.{ext}", dpi=200)

    print(f"Patient {args.instance}: true {y_test[args.instance]}, predicted {y_hat} (p = {p.max():.3f})")
    print(f"Anchors rule (estimated precision {anchors.last['precision']:.3f}): "
          f"IF {anchors.last['rule']} THEN {'readmitted' if y_hat else 'not readmitted'}")
    for (title, S, _), ok in zip(columns, sufficient):
        print(f"  {title.replace(chr(10), ' '):28s} {len(S):2d} features, formally sufficient: {ok}")
    print(f"Written: {out / 'tabular_patient.pdf'} (and .png)")


if __name__ == "__main__":
    main()

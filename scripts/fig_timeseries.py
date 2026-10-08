

import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xaibench.utils import RESULTS_DIR  # noqa: E402

METHODS = [("integrated_gradients", "Integrated Gradients"),
           ("gradient_shap", "GradientSHAP"),
           ("lime", "LIME (2-day blocks)")]
LABELS = {"MaxT": "Max. temperature", "MinT": "Min. temperature", "RH1": "Humidity (morning)",
          "RH2": "Humidity (afternoon)", "Wind": "Wind speed", "Rain": "Rainfall",
          "SSH": "Sunshine hours", "Evap": "Evaporation", "Radiation": "Solar radiation",
          "FAO56_ET": "Evapotranspiration"}


def normalise(a):
    m = np.abs(a).max()
    return a / m if m > 0 else a


def main():
    folder = RESULTS_DIR / "timeseries"
    files = sorted(glob.glob(str(folder / "maps_seed*.npz")))
    if not files:
        raise SystemExit(f"No maps_seed*.npz in {folder}: run the time-series experiment first.")
    maps = [np.load(f) for f in files]
    variables = [str(v) for v in maps[0]["variables"]]
    W = maps[0]["example_input"].shape[0]
    first = pd.read_csv(folder / "seed0.csv").iloc[0]       # the example window (seed 0)
    pred = "rain" if int(first["y_pred"]) == 1 else "no rain"

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0), sharex=True, sharey=True,
                             constrained_layout=True)
    days = [f"t-{W - 1 - i}" if i < W - 1 else "t" for i in range(W)]
    for col, (key, title) in enumerate(METHODS):
        ex = normalise(maps[0][f"example_{key}"])
        mean = normalise(np.mean([m[f"mean_abs_{key}"] for m in maps], axis=0))
        im_top = axes[0, col].imshow(ex.T, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        im_bot = axes[1, col].imshow(mean.T, cmap="Greys", vmin=0, vmax=1, aspect="auto")
        axes[0, col].set_title(title, fontsize=9)
    for ax in axes.ravel():
        ticks = list(range(W - 1, -1, -2))[::-1]          # ..., t-4, t-2, t
        ax.set_xticks(ticks)
        ax.set_xticklabels([days[i] for i in ticks], fontsize=6, rotation=90)
        ax.set_yticks(range(len(variables)))
        ax.set_yticklabels([LABELS.get(v, v) for v in variables], fontsize=7)
        ax.tick_params(length=2)
    axes[0, 0].set_ylabel(f"One window (predicted: {pred},\np = {first['p_pred']:.2f}, "
                          f"day t+1 = {first['target_date']})", fontsize=7)
    axes[1, 0].set_ylabel(f"Mean |attribution|\n({len(maps)} seeds x 200 windows)", fontsize=7)
    c1 = fig.colorbar(im_top, ax=axes[0, :], shrink=0.85, pad=0.01)
    c1.set_label("normalised attribution", fontsize=7); c1.ax.tick_params(labelsize=6)
    c2 = fig.colorbar(im_bot, ax=axes[1, :], shrink=0.85, pad=0.01)
    c2.set_label("normalised |attribution|", fontsize=7); c2.ax.tick_params(labelsize=6)

    out = RESULTS_DIR / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"timeseries_explanations.{ext}", dpi=200)
    print(f"Example window: day t+1 = {first['target_date']}, true = {int(first['y_true'])}, "
          f"predicted = {int(first['y_pred'])} (p = {first['p_pred']:.3f})")
    print(f"Written: {out / 'timeseries_explanations.pdf'} (and .png)")


if __name__ == "__main__":
    main()

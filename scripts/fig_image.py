

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import find_boundaries

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xaibench.utils import RESULTS_DIR  # noqa: E402

METHODS = [("cam", "CAM"), ("grad_cam", "Grad-CAM"), ("lime", "LIME"), ("kernel_shap", "KernelSHAP")]
K = 5


def main():
    path = RESULTS_DIR / "image" / "maps_seed0.npz"
    if not path.exists():
        raise SystemExit(f"{path} not found: run the image experiment first.")
    data = np.load(path, allow_pickle=True)
    images, segments, y, p = data["images"], data["segments"], data["y"], data["p"]
    rows = list(np.flatnonzero(y == 0)[:2]) + list(np.flatnonzero(y == 1)[:2])

    fig, axes = plt.subplots(len(rows), 1 + len(METHODS), figsize=(7.2, 1.55 * len(rows) + 0.4),
                             constrained_layout=True)
    cmap = plt.get_cmap("RdBu_r")
    for r, i in enumerate(rows):
        img, seg = images[i], segments[i]
        axes[r, 0].imshow(img)
        axes[r, 0].set_ylabel(f"{'dog' if y[i] else 'cat'}, p = {p[i]:.2f}", fontsize=7)
        for c, (key, title) in enumerate(METHODS, start=1):
            s = np.asarray(data[f"scores_{key}"][i], dtype=float)
            m = np.abs(s).max() or 1.0
            heat = cmap(0.5 + 0.5 * (s / m)[seg])[..., :3]
            overlay = 0.45 * img / 255.0 + 0.55 * heat
            top = np.argsort(-np.abs(s), kind="stable")[:K]
            border = find_boundaries(np.where(np.isin(seg, top), 1, 0), mode="thick")
            overlay[border] = (0.0, 0.0, 0.0)
            axes[r, c].imshow(overlay)
            if r == 0:
                axes[r, c].set_title(title, fontsize=8)
        if r == 0:
            axes[r, 0].set_title("Image", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    out = RESULTS_DIR / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"image_explanations.{ext}", dpi=200)
    print(f"Images shown (index in maps_seed0.npz): {rows}")
    print(f"Written: {out / 'image_explanations.pdf'} (and .png)")


if __name__ == "__main__":
    main()

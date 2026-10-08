
from __future__ import annotations

import io
import json
import time
import warnings
import zipfile
from dataclasses import asdict, dataclass
from typing import NamedTuple

import numpy as np

from xaibench.metrics import (EvalConfig, Task, evaluate_instance,
                              make_random_explainer)
from xaibench.utils import (DATA_DIR, RESULTS_DIR, ROOT, ResultsWriter,
                            get_logger, run_metadata, set_seed)

warnings.filterwarnings("ignore")
log = get_logger("image")

WEIGHTS = ROOT / "models" / "torch" / "hub" / "checkpoints" / "resnet18-f37072fd.pth"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)    # ImageNet statistics
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SIZE = 224


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ImageConfig:
    n_instances: int = 200          # upper bound; the test set has ~140 images
    n_segments: int = 50            # SLIC superpixels per image
    compactness: float = 20.0      # 10 gave long, irregular superpixels on real photos
    # fine-tuning
    epochs: int = 8
    batch_size: int = 32
    lr: float = 1e-4
    # metrics (Table 4 of the paper)
    k: int = 5
    eps_stability: float = 0.01     # Gaussian noise, fraction of the pixel range
    eps_robustness: float = 0.05
    n_perturbations: int = 10
    max_tries: int = 60
    n_background: int = 100
    # explainers
    lime_samples: int = 1000
    shap_samples: int = 1000

    @classmethod
    def smoke(cls) -> "ImageConfig":
        return cls(n_instances=4, epochs=1, n_perturbations=3, max_tries=20,
                   n_background=20, lime_samples=100, shap_samples=100)


class Img(NamedTuple):
    img: np.ndarray
    seg: np.ndarray


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_cats_dogs():
    from PIL import Image
    path = DATA_DIR / "raw" / "cats-and-dogs-image-classification.zip"
    images, labels, split = [], [], []
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            parts = name.lower().split("/")
            if parts[0] not in ("train", "test"):
                continue
            im = Image.open(io.BytesIO(z.read(name))).convert("RGB")
            w, h = im.size
            s = SIZE / min(w, h)
            im = im.resize((max(SIZE, round(w * s)), max(SIZE, round(h * s))), Image.BILINEAR)
            w, h = im.size
            left, top = (w - SIZE) // 2, (h - SIZE) // 2
            images.append(np.asarray(im.crop((left, top, left + SIZE, top + SIZE)), dtype=np.uint8))
            labels.append(int("dog" in parts[1]))
            split.append(parts[0])
    images, labels, split = np.stack(images), np.array(labels), np.array(split)
    log.info("Cats and Dogs: %d images (train %d, test %d), %.1f%% dogs", len(images),
             (split == "train").sum(), (split == "test").sum(), 100 * labels.mean())
    return images, labels, split


def superpixels(img: np.ndarray, n_segments: int, compactness: float) -> np.ndarray:
    """SLIC superpixels, labels 0..d-1."""
    from skimage.segmentation import slic
    seg = slic(img, n_segments=n_segments, compactness=compactness, start_label=0)
    return np.unique(seg, return_inverse=True)[1].reshape(seg.shape)


def mean_colour_image(x: Img) -> np.ndarray:
    d = x.seg.max() + 1
    counts = np.bincount(x.seg.ravel(), minlength=d)
    out = np.empty_like(x.img)
    for c in range(3):
        means = np.bincount(x.seg.ravel(), weights=x.img[..., c].ravel(), minlength=d) / counts
        out[..., c] = np.round(means[x.seg]).astype(np.uint8)
    return out


def per_superpixel(pixel_map: np.ndarray, seg: np.ndarray) -> np.ndarray:
    d = seg.max() + 1
    return (np.bincount(seg.ravel(), weights=pixel_map.ravel(), minlength=d)
            / np.bincount(seg.ravel(), minlength=d))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def to_tensor(images: np.ndarray, device):
    import torch
    x = (images.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.as_tensor(x.transpose(0, 3, 1, 2).copy(), device=device)


def build_model(device):
    import torch
    import torchvision
    model = torchvision.models.resnet18()
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    return model.to(device)


def fine_tune(model, images, y, cfg: ImageConfig, seed: int, device):
    import torch
    torch.manual_seed(seed)
    model.fc.reset_parameters()                     # the last layer depends on the seed
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    for epoch in range(cfg.epochs):
        model.train()
        perm = rng.permutation(len(images))
        total = 0.0
        for i in range(0, len(perm), cfg.batch_size):
            b = perm[i:i + cfg.batch_size]
            batch = images[b].copy()
            flip = rng.random(len(b)) < 0.5
            batch[flip] = batch[flip][:, :, ::-1]
            out = model(to_tensor(batch, device))
            loss = loss_fn(out, torch.as_tensor(y[b], device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        log.info("epoch %d  loss %.4f", epoch + 1, total / len(images))
    model.eval()
    return model


def make_predictor(model, device, batch_size: int = 128):
    import torch

    def predict(inputs):
        arr = np.stack([x.img if isinstance(x, Img) else x for x in inputs])
        out = []
        with torch.no_grad():
            for i in range(0, len(arr), batch_size):
                logits = model(to_tensor(arr[i:i + batch_size], device))
                out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 2))
    return predict


# ---------------------------------------------------------------------------
# Task (features = superpixels)
# ---------------------------------------------------------------------------

def make_task(predict, train_images, rng_background) -> Task:

    def remove(x: Img, S):
        out = x.img.copy()
        mask = np.isin(x.seg, list(S))
        out[mask] = mean_colour_image(x)[mask]
        return Img(out, x.seg)

    def complete(x: Img, S, z):
        out = (z.img if isinstance(z, Img) else z).copy()
        mask = np.isin(x.seg, list(S))
        out[mask] = x.img[mask]
        return Img(out, x.seg)

    def perturb(x: Img, eps, rng):
        noisy = x.img.astype(np.float32) + rng.normal(0.0, eps * 255.0, size=x.img.shape)
        return Img(np.clip(np.round(noisy), 0, 255).astype(np.uint8), x.seg)

    def background(rng, m):
        idx = rng_background.choice(len(train_images), size=m, replace=False)
        return [train_images[i] for i in idx]

    return Task(predict, remove, complete, perturb, background)


# ---------------------------------------------------------------------------
# Explainers: explain(x, seed) -> one score per superpixel
# (towards the predicted class)
# ---------------------------------------------------------------------------

def make_cam(model, device):
    import torch
    import torch.nn.functional as F
    feats = {}
    model.layer4.register_forward_hook(lambda m, i, o: feats.__setitem__("a", o))

    def explain(x: Img, seed: int = 0):
        with torch.no_grad():
            logits = model(to_tensor(x.img[None], device))
            target = int(logits.argmax())
            cam = torch.einsum("c,chw->hw", model.fc.weight[target], feats["a"][0])
            cam = F.interpolate(cam[None, None], size=x.seg.shape, mode="bilinear",
                                align_corners=False)[0, 0]
        return per_superpixel(cam.float().cpu().numpy(), x.seg)
    return explain


def make_grad_cam(model, device):
    import torch
    from captum.attr import LayerAttribution, LayerGradCam
    gc = LayerGradCam(model, model.layer4)

    def explain(x: Img, seed: int = 0):
        inp = to_tensor(x.img[None], device)
        with torch.no_grad():
            target = int(model(inp).argmax())
        attr = gc.attribute(inp, target=target, relu_attributions=True)
        attr = LayerAttribution.interpolate(attr, x.seg.shape, interpolate_mode="bilinear")
        return per_superpixel(attr[0, 0].detach().float().cpu().numpy(), x.seg)
    return explain


def make_kernel_shap(model, n_samples: int, device):
    import torch
    from captum.attr import KernelShap

    def forward(inp):
        return torch.softmax(model(inp), dim=1)
    ks = KernelShap(forward)

    def explain(x: Img, seed: int = 0):
        torch.manual_seed(seed)
        inp = to_tensor(x.img[None], device)
        base = to_tensor(mean_colour_image(x)[None], device)
        mask = torch.as_tensor(x.seg[None, None], device=device, dtype=torch.long)
        with torch.no_grad():
            target = int(forward(inp).argmax())
        attr = ks.attribute(inp, baselines=base, target=target, feature_mask=mask,
                            n_samples=n_samples, perturbations_per_eval=64)
        return per_superpixel(attr[0, 0].detach().float().cpu().numpy(), x.seg)
    return explain


def make_lime(predict, n_samples: int):
    from lime.lime_base import LimeBase
    from sklearn.metrics import pairwise_distances

    def kernel(d, width=0.25):
        return np.sqrt(np.exp(-(d ** 2) / width ** 2))

    def explain(x: Img, seed: int = 0):
        rng = np.random.default_rng(seed)
        d = x.seg.max() + 1
        masks = rng.integers(0, 2, size=(n_samples, d))
        masks[0, :] = 1
        fudge = mean_colour_image(x)
        probs = []
        for i in range(0, n_samples, 100):
            off = masks[i:i + 100][:, x.seg] == 0                 # (b, H, W)
            batch = np.where(off[..., None], fudge[None], x.img[None])
            probs.append(predict(list(batch)))
        probs = np.concatenate(probs)
        target = int(probs[0].argmax())
        distances = pairwise_distances(masks, masks[:1], metric="cosine").ravel()
        base = LimeBase(kernel, random_state=seed)
        _, weights, _, _ = base.explain_instance_with_data(
            masks, probs, distances, target, num_features=d, feature_selection="none")
        scores = np.zeros(d)
        for j, w in weights:
            scores[j] = w
        return scores
    return explain


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(seed: int = 0, smoke: bool = False) -> None:
    import torch
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
    cfg = ImageConfig.smoke() if smoke else ImageConfig()
    rng = set_seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = RESULTS_DIR / "image" / ("smoke" if smoke else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = ResultsWriter(out_dir / f"seed{seed}.csv")
    meta = run_metadata("image", seed)
    log.info("Config: %s | device: %s", asdict(cfg), device)

    # ---- data and ONE model ------------------------------------------------------
    images, y, split = load_cats_dogs()
    tr, te = np.flatnonzero(split == "train"), np.flatnonzero(split == "test")
    model = build_model(device)
    t0 = time.perf_counter()
    model = fine_tune(model, images[tr], y[tr], cfg, seed, device)
    log.info("Fine-tuning done in %.0f s", time.perf_counter() - t0)
    predict = make_predictor(model, device)
    proba = predict(list(images[te]))
    perf = {"accuracy": accuracy_score(y[te], proba.argmax(1)),
            "balanced_accuracy": balanced_accuracy_score(y[te], proba.argmax(1)),
            "auc": roc_auc_score(y[te], proba[:, 1]),
            "n_train": len(tr), "n_test": len(te), **meta, "config": asdict(cfg)}
    with open(out_dir / f"model_seed{seed}.json", "w") as f:
        json.dump(perf, f, indent=2, default=str)
    log.info("Model: accuracy %.3f | balanced accuracy %.3f | AUC %.3f (%d test images)",
             perf["accuracy"], perf["balanced_accuracy"], perf["auc"], len(te))

    # ---- task and explainers -------------------------------------------------------
    task = make_task(predict, images[tr], np.random.default_rng(seed + 1000))
    ecfg = EvalConfig(k=cfg.k, eps_stability=cfg.eps_stability, eps_robustness=cfg.eps_robustness,
                      n_perturbations=cfg.n_perturbations, max_tries=cfg.max_tries,
                      n_background=cfg.n_background)
    methods = {
        "cam": make_cam(model, device),
        "grad_cam": make_grad_cam(model, device),
        "lime": make_lime(predict, cfg.lime_samples),
        "kernel_shap": make_kernel_shap(model, cfg.shap_samples, device),
        "random": make_random_explainer(lambda x: int(x.seg.max()) + 1,
                                        np.random.default_rng(seed + 2000)),
    }

    # ---- explain every correctly classified test image (balanced if more than n) ----
    pred = proba.argmax(1)
    chosen = []
    for c in (0, 1):
        pool = [j for j in range(len(te)) if pred[j] == c and y[te[j]] == c]
        chosen += list(rng.permutation(pool)[:cfg.n_instances // 2])
    chosen = sorted(chosen)
    log.info("Explaining %d correctly classified test images with %s", len(chosen), list(methods))
    saved = {"images": [], "segments": [], "y": [], "p": []}
    saved.update({f"scores_{n}": [] for n in methods})
    for n_done, j in enumerate(chosen, start=1):
        img = images[te[j]]
        x = Img(img, superpixels(img, cfg.n_segments, cfg.compactness))
        p = proba[j]
        for name, explain in methods.items():
            t0 = time.perf_counter()
            scores = np.asarray(explain(x, seed), dtype=float)
            runtime = time.perf_counter() - t0
            res = evaluate_instance(task, explain, x, ecfg, rng, seed=seed, k=cfg.k,
                                    scores=scores, runtime=runtime)
            saved[f"scores_{name}"].append(scores)
            writer.write({**meta, "instance": int(te[j]), "y_true": int(y[te[j]]),
                          "y_pred": int(p.argmax()), "p_pred": float(p.max()),
                          "method": name, "kind": "attribution", **res,
                          "n_superpixels": int(x.seg.max()) + 1})
        saved["images"].append(img); saved["segments"].append(x.seg)
        saved["y"].append(int(y[te[j]])); saved["p"].append(float(p.max()))
        log.info("[%d/%d] test image %d (%s, %d superpixels) done", n_done, len(chosen), te[j],
                 "dog" if y[te[j]] else "cat", x.seg.max() + 1)

    np.savez_compressed(out_dir / f"maps_seed{seed}.npz",
                        images=np.stack(saved.pop("images")),
                        segments=np.stack(saved.pop("segments")),
                        y=np.array(saved.pop("y")), p=np.array(saved.pop("p")),
                        **{k: np.array(v, dtype=object) for k, v in saved.items()})
    log.info("Results written to %s", writer.path)

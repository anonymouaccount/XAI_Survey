

from __future__ import annotations

import json
import re
import time
import warnings
import zipfile
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from xaibench.metrics import (EvalConfig, Task, evaluate_instance,
                              make_random_explainer, top_k)
from xaibench.utils import (DATA_DIR, RESULTS_DIR, ROOT, ResultsWriter,
                            get_logger, run_metadata, set_seed)

warnings.filterwarnings("ignore")
log = get_logger("text")

MODEL_DIR = ROOT / "models" / "hf" / "distilbert-base-uncased"
MASK = "[MASK]"                        # a removed word


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TextConfig:
    n_instances: int = 200
    max_len: int = 256              # tokens, including [CLS] and [SEP]
    min_words: int = 20
    # fine-tuning
    epochs: int = 2
    batch_size: int = 32
    lr: float = 2e-5
    warmup_frac: float = 0.1
    n_train: int | None = None      # None = the whole training half
    n_test: int | None = None
    # metrics (Table 4 of the paper)
    k: int = 10
    eps_stability: float = 1        # number of replaced words
    eps_robustness: float = 3
    n_perturbations: int = 10
    max_tries: int = 60
    n_background: int = 100
    n_neighbours: int = 10
    # explainers
    ig_steps: int = 50
    lime_samples: int = 1000

    @classmethod
    def smoke(cls) -> "TextConfig":
        return cls(n_instances=4, epochs=1, n_train=2000, n_test=1000, n_perturbations=3,
                   max_tries=20, n_background=20, ig_steps=10, lime_samples=100)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def clean(text: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_imdb():
    path = DATA_DIR / "raw" / "imdb-dataset-of-50k-movie-reviews.zip"
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        with z.open(name) as f:
            df = pd.read_csv(f)
    n_raw = len(df)
    df["review"] = df["review"].map(clean)
    df = df.drop_duplicates("review").reset_index(drop=True)
    y = (df["sentiment"].str.lower() == "positive").astype(int).to_numpy()
    log.info("IMDB: %d reviews, %d after removing duplicates, %.1f%% positive",
             n_raw, len(df), 100 * y.mean())
    return df["review"].tolist(), y


# ---------------------------------------------------------------------------
# Words -> token ids
# ---------------------------------------------------------------------------

class WordEncoder:

    def __init__(self, tokenizer, max_len: int):
        self.tok = tokenizer
        self.max_len = max_len
        self.cache = {}
        self.cls, self.sep = tokenizer.cls_token_id, tokenizer.sep_token_id
        self.pad, self.mask = tokenizer.pad_token_id, tokenizer.mask_token_id

    def word_ids(self, word: str):
        if word == MASK:
            return [self.mask]
        ids = self.cache.get(word)
        if ids is None:
            ids = self.tok(word, add_special_tokens=False)["input_ids"] or [self.tok.unk_token_id]
            self.cache[word] = ids
        return ids

    def encode(self, words):
        ids, owner = [self.cls], [-1]
        for i, w in enumerate(words):
            piece = self.word_ids(w)
            if len(ids) + len(piece) > self.max_len - 1:
                break
            ids += piece
            owner += [i] * len(piece)
        return ids + [self.sep], owner + [-1]

    def n_tokens(self, words) -> int:
        return 2 + sum(len(self.word_ids(w)) for w in words)

    def batch(self, list_of_words, device):
        import torch
        encoded = [self.encode(w)[0] for w in list_of_words]
        L = max(len(e) for e in encoded)
        ids = torch.full((len(encoded), L), self.pad, dtype=torch.long)
        att = torch.zeros((len(encoded), L), dtype=torch.long)
        for i, e in enumerate(encoded):
            ids[i, :len(e)] = torch.tensor(e)
            att[i, :len(e)] = 1
        return ids.to(device), att.to(device)


def make_predictor(model, encoder: WordEncoder, device, batch_size: int = 64):
    import torch

    def predict(list_of_words):
        model.eval()
        out = []
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                             enabled=device.type == "cuda"):
            for i in range(0, len(list_of_words), batch_size):
                ids, att = encoder.batch(list_of_words[i:i + batch_size], device)
                logits = model(input_ids=ids, attention_mask=att).logits.float()
                out.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 2))
    return predict


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def fine_tune(texts, y, cfg: TextConfig, seed: int, device):
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=2).to(device)
    enc = tokenizer(texts, truncation=True, max_length=cfg.max_len, padding="max_length",
                    return_tensors="pt")
    labels = torch.as_tensor(y)
    n = len(texts)
    steps = cfg.epochs * ((n + cfg.batch_size - 1) // cfg.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_frac * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    gen = torch.Generator().manual_seed(seed)
    step = 0
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, cfg.batch_size):
            b = perm[i:i + cfg.batch_size]
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                out = model(input_ids=enc["input_ids"][b].to(device),
                            attention_mask=enc["attention_mask"][b].to(device),
                            labels=labels[b].to(device))
            opt.zero_grad()
            scaler.scale(out.loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 100 == 0:
                log.info("epoch %d  step %d/%d  loss %.4f", epoch + 1, step, steps, out.loss.item())
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Perturbations: nearest neighbours in the embedding space of the model
# ---------------------------------------------------------------------------

def embedding_neighbours(model, tokenizer, n_neighbours: int, device):
    import torch
    vocab = tokenizer.get_vocab()
    words = [w for w in vocab if w.isalpha() and len(w) >= 2 and not w.startswith("##")]
    ids = torch.tensor([vocab[w] for w in words], device=device)
    E = model.get_input_embeddings().weight.detach()[ids].float()
    E = E / E.norm(dim=1, keepdim=True)
    neighbours = {}
    for i in range(0, len(words), 2048):
        sim = E[i:i + 2048] @ E.T
        sim[torch.arange(sim.shape[0]), torch.arange(i, i + sim.shape[0])] = -1   # not itself
        best = sim.topk(n_neighbours, dim=1).indices.cpu().numpy()
        for r, row in enumerate(best):
            neighbours[words[i + r]] = [words[j] for j in row]
    log.info("Embedding neighbours computed for %d whole-word tokens", len(neighbours))
    return neighbours


# ---------------------------------------------------------------------------
# Task (features = words)
# ---------------------------------------------------------------------------

def make_task(predict, train_words, neighbours, rng_background) -> Task:

    def remove(x, S):
        return tuple(MASK if i in S else w for i, w in enumerate(x))

    def complete(x, S, z):
        # keep the words of S; every other position takes the word of another
        # (training) review at the same position
        return tuple(w if i in S else z[i % len(z)] for i, w in enumerate(x))

    def perturb(x, eps, rng):
        x = list(x)
        eligible = [i for i, w in enumerate(x) if w.lower() in neighbours]
        n = min(int(round(eps)), len(eligible))
        for i in rng.choice(eligible, size=n, replace=False) if n else []:
            x[i] = str(rng.choice(neighbours[x[i].lower()]))
        return tuple(x)

    def background(rng, m):
        idx = rng_background.choice(len(train_words), size=m, replace=False)
        return [train_words[i] for i in idx]

    return Task(predict, remove, complete, perturb, background)


# ---------------------------------------------------------------------------
# Explainers: explain(words, seed) -> one score per word
# (attributions to the probability of the predicted class)
# ---------------------------------------------------------------------------

def make_integrated_gradients(model, encoder: WordEncoder, n_steps: int, device):
    import torch
    from captum.attr import LayerIntegratedGradients

    def forward(input_ids, attention_mask):
        return torch.softmax(model(input_ids=input_ids, attention_mask=attention_mask).logits, dim=1)

    lig = LayerIntegratedGradients(forward, model.get_input_embeddings())

    def explain(words, seed: int = 0):
        ids, owner = encoder.encode(words)
        ids_t = torch.tensor([ids], device=device)
        att = torch.ones_like(ids_t)
        base = ids_t.clone()
        base[0, 1:-1] = encoder.mask                   # every word masked = removal baseline
        with torch.no_grad():
            target = int(forward(ids_t, att).argmax())
        attr = lig.attribute(ids_t, baselines=base, additional_forward_args=(att,),
                             target=target, n_steps=n_steps, internal_batch_size=25)
        token_scores = attr.sum(dim=-1)[0].detach().float().cpu().numpy()
        scores = np.zeros(len(words))
        for t, w in enumerate(owner):
            if w >= 0:
                scores[w] += token_scores[t]
        return scores
    return explain


def make_lime(predict, n_samples: int):
    from lime.lime_base import LimeBase
    from sklearn.metrics import pairwise_distances

    def kernel(d, width=25.0):
        return np.sqrt(np.exp(-(d ** 2) / width ** 2))

    def explain(words, seed: int = 0):
        rng = np.random.default_rng(seed)
        d = len(words)
        masks = np.ones((n_samples, d), dtype=int)
        for r in range(1, n_samples):                 # row 0 = the review itself
            n_off = rng.integers(1, d)
            masks[r, rng.choice(d, size=n_off, replace=False)] = 0
        inputs = [tuple(w if keep else MASK for w, keep in zip(words, m)) for m in masks]
        probs = predict(inputs)
        target = int(probs[0].argmax())
        distances = pairwise_distances(masks, masks[:1], metric="cosine").ravel() * 100
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

def select_instances(predict, test_words, y_test, encoder, cfg, rng):
    fits = [i for i, w in enumerate(test_words)
            if len(w) >= cfg.min_words and encoder.n_tokens(w) <= cfg.max_len]
    candidates = rng.permutation(fits)[:20 * cfg.n_instances]
    proba = predict([test_words[i] for i in candidates])
    pred = proba.argmax(1)
    chosen = []
    for c in (0, 1):
        pool = [i for i, p in zip(candidates, pred) if p == c and y_test[i] == c]
        chosen += pool[:cfg.n_instances // 2]
    log.info("%d test reviews fit in %d tokens; %d selected", len(fits), cfg.max_len, len(chosen))
    return sorted(chosen)


def main(seed: int = 0, smoke: bool = False) -> None:
    import torch
    cfg = TextConfig.smoke() if smoke else TextConfig()
    rng = set_seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = RESULTS_DIR / "text" / ("smoke" if smoke else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = ResultsWriter(out_dir / f"seed{seed}.csv")
    meta = run_metadata("text", seed)
    log.info("Config: %s | device: %s", asdict(cfg), device)

    # ---- data and ONE model ------------------------------------------------------
    texts, y = load_imdb()
    tr_txt, te_txt, y_tr, y_te = train_test_split(texts, y, test_size=0.5, stratify=y,
                                                  random_state=seed)
    if cfg.n_train:
        tr_txt, y_tr = tr_txt[:cfg.n_train], y_tr[:cfg.n_train]
    if cfg.n_test:
        te_txt, y_te = te_txt[:cfg.n_test], y_te[:cfg.n_test]
    t0 = time.perf_counter()
    model, tokenizer = fine_tune(tr_txt, y_tr, cfg, seed, device)
    log.info("Fine-tuning done in %.0f s", time.perf_counter() - t0)

    encoder = WordEncoder(tokenizer, cfg.max_len)
    predict = make_predictor(model, encoder, device, batch_size=128)
    test_words = [tuple(t.split()) for t in te_txt]
    train_words = [tuple(t.split()) for t in tr_txt]
    proba = predict(test_words)
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
    fits = np.array([encoder.n_tokens(w) <= cfg.max_len for w in test_words])
    perf = {"accuracy": accuracy_score(y_te, proba.argmax(1)),
            "balanced_accuracy": balanced_accuracy_score(y_te, proba.argmax(1)),
            "auc": roc_auc_score(y_te, proba[:, 1]),
            "accuracy_reviews_fitting_input": accuracy_score(y_te[fits], proba[fits].argmax(1)),
            "share_reviews_fitting_input": float(fits.mean()),
            "n_train": len(tr_txt), "n_test": len(te_txt), **meta, "config": asdict(cfg)}
    with open(out_dir / f"model_seed{seed}.json", "w") as f:
        json.dump(perf, f, indent=2, default=str)
    log.info("Model: accuracy %.3f | AUC %.3f | accuracy on reviews that fit in %d tokens "
             "(%.0f%% of the test set): %.3f", perf["accuracy"], perf["auc"], cfg.max_len,
             100 * perf["share_reviews_fitting_input"], perf["accuracy_reviews_fitting_input"])

    # ---- task and explainers -------------------------------------------------------
    neighbours = embedding_neighbours(model, tokenizer, cfg.n_neighbours, device)
    task = make_task(predict, train_words, neighbours, np.random.default_rng(seed + 1000))
    ecfg = EvalConfig(k=cfg.k, eps_stability=cfg.eps_stability, eps_robustness=cfg.eps_robustness,
                      n_perturbations=cfg.n_perturbations, max_tries=cfg.max_tries,
                      n_background=cfg.n_background)
    methods = {
        "integrated_gradients": make_integrated_gradients(model, encoder, cfg.ig_steps, device),
        "lime": make_lime(predict, cfg.lime_samples),
        "random": make_random_explainer(lambda x: len(x), np.random.default_rng(seed + 2000)),
    }

    # ---- explain the same reviews with every method ----------------------------------
    chosen = select_instances(predict, test_words, y_te, encoder, cfg, rng)
    example = {}
    for n_done, j in enumerate(chosen, start=1):
        x = test_words[j]
        p = predict([x])[0]
        for name, explain in methods.items():
            t0 = time.perf_counter()
            scores = np.asarray(explain(x, seed), dtype=float)
            runtime = time.perf_counter() - t0
            res = evaluate_instance(task, explain, x, ecfg, rng, seed=seed, k=cfg.k,
                                    scores=scores, runtime=runtime)
            S = sorted(top_k(scores, cfg.k))
            if n_done == 1:
                example[name] = scores.tolist()
            writer.write({**meta, "instance": int(j), "n_words": len(x),
                          "y_true": int(y_te[j]), "y_pred": int(p.argmax()),
                          "p_pred": float(p.max()), "method": name, "kind": "attribution",
                          **res, "top_words": "|".join(x[i] for i in S)})
        if n_done == 1:
            example.update({"words": list(x), "y_true": int(y_te[j]), "y_pred": int(p.argmax()),
                            "p_pred": float(p.max())})
            with open(out_dir / f"example_seed{seed}.json", "w") as f:
                json.dump(example, f)
        log.info("[%d/%d] review %d (%d words) done", n_done, len(chosen), j, len(x))
    log.info("Results written to %s", writer.path)

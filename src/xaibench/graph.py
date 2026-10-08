

from __future__ import annotations

import json
import time
import warnings
from dataclasses import asdict, dataclass

import numpy as np

from xaibench.metrics import (EvalConfig, Task, evaluate_instance,
                              make_random_explainer, top_k)
from xaibench.utils import (RESULTS_DIR, ResultsWriter, get_logger,
                            run_metadata, set_seed)

warnings.filterwarnings("ignore")
log = get_logger("graph")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GraphConfig:
    n_instances: int = 200
    base_nodes: int = 300
    ba_edges: int = 5
    n_motifs: int = 80
    n_features: int = 10            # constant node features
    num_hops: int = 3
    # model
    hidden: int = 32
    epochs: int = 2000
    lr: float = 0.01
    weight_decay: float = 5e-4
    # metrics (Table 4 of the paper)
    k: int = 6                      # edges of a house
    eps_stability: float = 0.01     # fraction of non-motif edges deleted
    eps_robustness: float = 0.05
    n_perturbations: int = 10
    max_tries: int = 60
    n_background: int = 100
    # explainers
    gnnexplainer_epochs: int = 100
    pgexplainer_epochs: int = 30
    ig_steps: int = 50
    lime_samples: int = 1000

    @classmethod
    def smoke(cls) -> "GraphConfig":
        return cls(n_instances=4, epochs=300, n_perturbations=3, max_tries=20, n_background=20,
                   gnnexplainer_epochs=30, pgexplainer_epochs=3, ig_steps=10, lime_samples=100)


# ---------------------------------------------------------------------------
# Data and model
# ---------------------------------------------------------------------------

def make_ba_shapes(cfg: GraphConfig, seed: int):
    import torch
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from torch_geometric.datasets import ExplainerDataset
    from torch_geometric.datasets.graph_generator import BAGraph
    from torch_geometric.datasets.motif_generator import HouseMotif
    torch.manual_seed(seed)
    g = ExplainerDataset(graph_generator=BAGraph(num_nodes=cfg.base_nodes, num_edges=cfg.ba_edges),
                         motif_generator=HouseMotif(), num_motifs=cfg.n_motifs)[0]
    n = g.y.numel()
    # house id of every motif node: connected components of the motif edges
    ei = g.edge_index[:, g.edge_mask.bool()].numpy()
    comp = connected_components(coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)),
                                directed=False)[1]
    house = np.where(g.node_mask.numpy() > 0, comp, -1)
    x = torch.ones(n, cfg.n_features)
    log.info("BA-Shapes: %d nodes, %d undirected edges, %d houses, labels %s", n,
             g.edge_index.shape[1] // 2, len(set(house[house >= 0])),
             np.bincount(g.y.numpy()).tolist())
    return x, g.edge_index, g.y, house


def build_model(n_features: int, hidden: int, n_classes: int):
    import torch
    from torch_geometric.nn import GCNConv

    class GCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = torch.nn.ModuleList([GCNConv(n_features, hidden), GCNConv(hidden, hidden),
                                              GCNConv(hidden, hidden)])
            self.out = torch.nn.Linear(hidden, n_classes)

        def forward(self, x, edge_index, edge_weight=None, **kwargs):
            # edge_weight (one value per directed edge, default 1) is used by
            # Integrated Gradients: weight 0 = edge removed
            for conv in self.convs:
                x = torch.relu(conv(x, edge_index, edge_weight))
            return self.out(x)

    return GCN()


def train_model(model, x, edge_index, y, cfg, seed, device):
    import torch
    from sklearn.model_selection import train_test_split
    idx = np.arange(y.numel())
    tr, rest = train_test_split(idx, test_size=0.2, stratify=y.numpy(), random_state=seed)
    va, te = train_test_split(rest, test_size=0.5, stratify=y.numpy()[rest], random_state=seed)
    torch.manual_seed(seed)
    x, edge_index, y_d = x.to(device), edge_index.to(device), y.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best, best_state = -1.0, None
    for epoch in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        out = model(x, edge_index)
        loss = torch.nn.functional.cross_entropy(out[tr], y_d[tr])
        loss.backward()
        opt.step()
        if epoch % 10 == 0 or epoch == cfg.epochs - 1:
            model.eval()
            with torch.no_grad():
                pred = model(x, edge_index).argmax(1).cpu().numpy()
            acc_va = (pred[va] == y.numpy()[va]).mean()
            if acc_va > best:
                best = acc_va
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if epoch % 500 == 0:
                log.info("epoch %4d  loss %.4f  validation accuracy %.3f", epoch, loss.item(), acc_va)
    model.load_state_dict(best_state)
    model.eval()
    return model, tr, va, te


# ---------------------------------------------------------------------------
# An explained node and its 3-hop neighbourhood
# ---------------------------------------------------------------------------

class GraphInstance:

    def __init__(self, x, und, center, gt, keep=None, nodes=None):
        self.x, self.und, self.center, self.gt = x, und, center, gt
        self.keep = np.ones(len(und), dtype=bool) if keep is None else keep
        self.nodes = nodes

    @property
    def d(self):
        return len(self.und)

    def with_keep(self, keep):
        return GraphInstance(self.x, self.und, self.center, self.gt, keep, self.nodes)

    def edge_index(self, device):
        import torch
        ids = np.flatnonzero(self.keep)
        e = self.und[ids]
        ei = np.concatenate([e, e[:, ::-1]]).T
        return torch.as_tensor(ei.copy(), dtype=torch.long, device=device), np.concatenate([ids, ids])


def make_instance(node, x, edge_index, house, num_hops):
    from torch_geometric.utils import k_hop_subgraph
    subset, sub_ei, mapping, _ = k_hop_subgraph(int(node), num_hops, edge_index,
                                                relabel_nodes=True, num_nodes=x.shape[0])
    sub = sub_ei.numpy()
    und = sub[:, sub[0] < sub[1]].T.copy()                    # one column per undirected edge
    glob = subset.numpy()
    h = house[node]
    gt = (house[glob[und[:, 0]]] == h) & (house[glob[und[:, 1]]] == h)
    return GraphInstance(x[subset], und, int(mapping[0]), gt, nodes=glob)


def edge_scores(directed_scores, feature_ids, inst: GraphInstance):
    s = np.bincount(feature_ids, weights=directed_scores, minlength=inst.d)
    s[~inst.keep] = 0.0
    return s


# ---------------------------------------------------------------------------
# Predictions (batched as one disjoint union of graphs)
# ---------------------------------------------------------------------------

def make_predictor(model, device, batch_size: int = 100):
    import torch

    def predict(instances):
        out = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(instances), batch_size):
                chunk = instances[i:i + batch_size]
                xs, eis, centers, offset = [], [], [], 0
                for inst in chunk:
                    ei, _ = inst.edge_index(device)
                    xs.append(inst.x); eis.append(ei + offset)
                    centers.append(offset + inst.center); offset += inst.x.shape[0]
                logits = model(torch.cat(xs).to(device), torch.cat(eis, dim=1))
                out.append(torch.softmax(logits[centers], dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 4))
    return predict


# ---------------------------------------------------------------------------
# Task (features = undirected edges)
# ---------------------------------------------------------------------------

def make_task(predict) -> Task:

    def remove(inst, S):
        keep = inst.keep.copy()
        keep[list(S)] = False
        return inst.with_keep(keep)

    def complete(inst, S, z):
        # z is a random seed: every edge outside S is kept with probability 1/2
        keep = (np.random.default_rng(int(z)).random(inst.d) < 0.5) & inst.keep
        keep[list(S)] = True
        return inst.with_keep(keep)

    def perturb(inst, eps, rng):
        candidates = np.flatnonzero(inst.keep & ~inst.gt)
        n = min(len(candidates), max(1, int(round(eps * inst.d))))
        keep = inst.keep.copy()
        keep[rng.choice(candidates, size=n, replace=False)] = False
        return inst.with_keep(keep)

    def background(rng, m):
        return list(rng.integers(0, 2**31 - 1, size=m))

    return Task(predict, remove, complete, perturb, background)


# ---------------------------------------------------------------------------
# Explainers: explain(inst, seed) -> one score per undirected edge
# ---------------------------------------------------------------------------

MODEL_CONFIG = dict(mode="multiclass_classification", task_level="node", return_type="raw")


def make_pyg_explainer(model, algorithm, device, explanation_type="model"):
    """Wrap a PyG explanation algorithm that returns an edge mask."""
    import torch
    from torch_geometric.explain import Explainer
    explainer = Explainer(model=model, algorithm=algorithm, explanation_type=explanation_type,
                          node_mask_type=None, edge_mask_type="object", model_config=MODEL_CONFIG)

    def explain(inst: GraphInstance, seed: int = 0):
        torch.manual_seed(seed)
        ei, fid = inst.edge_index(device)
        x = inst.x.to(device)
        kwargs = {}
        if explanation_type == "phenomenon":
            with torch.no_grad():
                kwargs["target"] = model(x, ei).argmax(1)
        expl = explainer(x, ei, index=inst.center, **kwargs)
        return edge_scores(expl.edge_mask.detach().float().cpu().numpy(), fid, inst)
    return explainer, explain


def make_integrated_gradients(model, n_steps: int, device):
    import torch

    def explain(inst: GraphInstance, seed: int = 0):
        ei, fid = inst.edge_index(device)
        x = inst.x.to(device)
        with torch.no_grad():
            target = int(model(x, ei)[inst.center].argmax())
        grads = torch.zeros(ei.shape[1], device=device)
        for a in (torch.arange(n_steps, dtype=torch.float32) + 0.5) / n_steps:
            w = torch.full((ei.shape[1],), float(a), device=device, requires_grad=True)
            prob = torch.softmax(model(x, ei, w)[inst.center], dim=0)[target]
            grads += torch.autograd.grad(prob, w)[0]
        return edge_scores((grads / n_steps).cpu().numpy(), fid, inst)
    return explain


def train_pgexplainer(explainer, model, x, edge_index, nodes, epochs, seed, device):
    import torch
    torch.manual_seed(seed)
    x, edge_index = x.to(device), edge_index.to(device)
    with torch.no_grad():
        target = model(x, edge_index).argmax(1)
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        total = 0.0
        for idx in rng.permutation(nodes):
            total += explainer.algorithm.train(epoch, model, x, edge_index, target=target,
                                               index=int(idx))
        if epoch % 10 == 0 or epoch == epochs - 1:
            log.info("PGExplainer epoch %d  loss %.4f", epoch, total / len(nodes))


def make_lime(predict, n_samples: int):
    from lime.lime_base import LimeBase
    from sklearn.metrics import pairwise_distances

    def kernel(d, width=0.25):
        return np.sqrt(np.exp(-(d ** 2) / width ** 2))

    def explain(inst: GraphInstance, seed: int = 0):
        rng = np.random.default_rng(seed)
        present = np.flatnonzero(inst.keep)
        m = len(present)
        masks = rng.integers(0, 2, size=(n_samples, m))
        masks[0, :] = 1
        graphs = []
        for row in masks:
            keep = np.zeros(inst.d, dtype=bool)
            keep[present[row == 1]] = True
            graphs.append(inst.with_keep(keep))
        probs = predict(graphs)
        target = int(probs[0].argmax())
        distances = pairwise_distances(masks, masks[:1], metric="cosine").ravel()
        base = LimeBase(kernel, random_state=seed)
        _, weights, _, _ = base.explain_instance_with_data(
            masks, probs, distances, target, num_features=m, feature_selection="none")
        directed = np.zeros(m)
        for j, w in weights:
            directed[j] = w
        s = np.zeros(inst.d)
        s[present] = directed
        return edge_scores(s, np.arange(inst.d), inst)
    return explain


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(seed: int = 0, smoke: bool = False) -> None:
    import torch
    from sklearn.metrics import roc_auc_score
    from torch_geometric.explain.algorithm import GNNExplainer, PGExplainer
    cfg = GraphConfig.smoke() if smoke else GraphConfig()
    rng = set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = RESULTS_DIR / "graph" / ("smoke" if smoke else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = ResultsWriter(out_dir / f"seed{seed}.csv")
    meta = run_metadata("graph", seed)
    log.info("Config: %s | device: %s", asdict(cfg), device)

    # ---- data and ONE model ------------------------------------------------------
    x, edge_index, y, house = make_ba_shapes(cfg, seed)
    model = build_model(cfg.n_features, cfg.hidden, int(y.max()) + 1).to(device)
    model, tr, va, te = train_model(model, x, edge_index, y, cfg, seed, device)
    with torch.no_grad():
        pred = model(x.to(device), edge_index.to(device)).argmax(1).cpu().numpy()
    yn = y.numpy()
    perf = {"accuracy": float((pred[te] == yn[te]).mean()),
            "train_accuracy": float((pred[tr] == yn[tr]).mean()),
            "validation_accuracy": float((pred[va] == yn[va]).mean()),
            "n_nodes": int(yn.size), "n_edges": int(edge_index.shape[1] // 2),
            "n_train": len(tr), "n_val": len(va), "n_test": len(te), **meta, "config": asdict(cfg)}
    with open(out_dir / f"model_seed{seed}.json", "w") as f:
        json.dump(perf, f, indent=2, default=str)
    log.info("Model: test accuracy %.3f (train %.3f)", perf["accuracy"], perf["train_accuracy"])

    # ---- explainers ----------------------------------------------------------------
    predict = make_predictor(model, device)
    task = make_task(predict)
    ecfg = EvalConfig(k=cfg.k, eps_stability=cfg.eps_stability, eps_robustness=cfg.eps_robustness,
                      n_perturbations=cfg.n_perturbations, max_tries=cfg.max_tries,
                      n_background=cfg.n_background)
    _, gnnexp = make_pyg_explainer(model, GNNExplainer(epochs=cfg.gnnexplainer_epochs), device)
    pg_explainer, pgexp = make_pyg_explainer(
        model, PGExplainer(epochs=cfg.pgexplainer_epochs, lr=0.003).to(device), device,
        explanation_type="phenomenon")
    motif_train = [i for i in tr if yn[i] > 0]
    t0 = time.perf_counter()
    train_pgexplainer(pg_explainer, model, x, edge_index, motif_train, cfg.pgexplainer_epochs,
                      seed, device)
    log.info("PGExplainer trained in %.0f s", time.perf_counter() - t0)
    igexp = make_integrated_gradients(model, cfg.ig_steps, device)
    methods = {"gnnexplainer": gnnexp, "pgexplainer": pgexp,
               "integrated_gradients": igexp, "lime": make_lime(predict, cfg.lime_samples),
               "random": make_random_explainer(lambda inst: inst.d, np.random.default_rng(seed + 2000))}

    # ---- explain correctly classified motif nodes --------------------------------------
    pool = [i for i in range(yn.size) if yn[i] > 0 and pred[i] == yn[i]]
    chosen = sorted(rng.permutation(pool)[:cfg.n_instances])
    log.info("Explaining %d of %d correctly classified motif nodes with %s",
             len(chosen), len(pool), list(methods))
    example = None
    for n_done, node in enumerate(chosen, start=1):
        inst = make_instance(node, x, edge_index, house, cfg.num_hops)
        p = predict([inst])[0]
        node_scores = {}
        for name, explain in methods.items():
            t0 = time.perf_counter()
            scores = np.asarray(explain(inst, seed), dtype=float)
            runtime = time.perf_counter() - t0
            res = evaluate_instance(task, explain, inst, ecfg, rng, seed=seed, k=cfg.k,
                                    scores=scores, runtime=runtime)
            S = list(top_k(scores, cfg.k))
            # importance = |score|, as in top_k (a strongly negative edge is important too);
            # the signed AUC is kept for reference
            valid = 0 < inst.gt.sum() < inst.d
            auc_gt = roc_auc_score(inst.gt, np.abs(scores)) if valid else float("nan")
            auc_signed = roc_auc_score(inst.gt, scores) if valid else float("nan")
            writer.write({**meta, "instance": int(node), "y_true": int(yn[node]),
                          "y_pred": int(p.argmax()), "p_pred": float(p.max()),
                          "method": name, "kind": "attribution", **res,
                          "auc_gt": auc_gt, "auc_gt_signed": auc_signed,
                          "precision_gt": float(inst.gt[S].mean()),
                          "n_gt_edges": int(inst.gt.sum())})
            node_scores[name] = scores.tolist()
        # example for the figure: the first node with a medium neighbourhood
        # (20-80 edges), otherwise the last explained node
        if example is None and (20 <= inst.d <= 80 or n_done == len(chosen)):
            example = {"node": int(node), "label": int(yn[node]), "p_pred": float(p.max()),
                       "nodes": inst.nodes.tolist(), "center": inst.center,
                       "edges": inst.und.tolist(), "gt": inst.gt.tolist(), **node_scores}
            with open(out_dir / f"example_seed{seed}.json", "w") as f:
                json.dump(example, f)
        log.info("[%d/%d] node %d (label %d, %d edges in its 3-hop neighbourhood) done",
                 n_done, len(chosen), node, yn[node], inst.d)
    log.info("Results written to %s", writer.path)

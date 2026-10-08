

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xaibench.utils import RESULTS_DIR  # noqa: E402

METHODS = [("gnnexplainer", "GNNExplainer"), ("pgexplainer", "PGExplainer"),
           ("integrated_gradients", "Integrated Gradients"), ("lime", "LIME")]
K = 6


def main():
    path = RESULTS_DIR / "graph" / "example_seed0.json"
    if not path.exists():
        raise SystemExit(f"{path} not found: run the graph experiment first.")
    ex = json.load(open(path))
    edges = [tuple(e) for e in ex["edges"]]
    gt = np.array(ex["gt"], dtype=bool)
    G = nx.Graph(); G.add_edges_from(edges)
    pos = nx.kamada_kawai_layout(G)
    house_nodes = {n for (a, b), g in zip(edges, gt) if g for n in (a, b)}

    fig, axes = plt.subplots(1, len(METHODS), figsize=(7.2, 2.3), constrained_layout=True)
    for ax, (key, title) in zip(axes, METHODS):
        s = np.abs(np.asarray(ex[key], dtype=float))
        s = s / (s.max() or 1.0)
        top = set(np.argsort(-s, kind="stable")[:K])
        nx.draw_networkx_edges(G, pos, edgelist=[e for e, g in zip(edges, gt) if g], ax=ax,
                               width=6, edge_color="#b7e4b0")
        nx.draw_networkx_edges(G, pos, edgelist=edges, ax=ax, width=0.3 + 2.2 * s,
                               edge_color=["#c0392b" if i in top else str(0.85 - 0.6 * v)
                                           for i, v in enumerate(s)])
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=6,
                               node_color=["#2e7d32" if n in house_nodes else "0.4" for n in G])
        nx.draw_networkx_nodes(G, pos, nodelist=[ex["center"]], ax=ax, node_size=60,
                               node_shape="*", node_color="black")
        hits = len(top & set(np.flatnonzero(gt)))
        ax.set_title(f"{title}\n{hits}/{K} top edges in the house", fontsize=7.5)
        ax.axis("off")
    out = RESULTS_DIR / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"graph_explanation.{ext}", dpi=200)
    print(f"Node {ex['node']} (label {ex['label']}): {len(edges)} edges, {gt.sum()} house edges")
    print(f"Written: {out / 'graph_explanation.pdf'} (and .png)")


if __name__ == "__main__":
    main()

"""Aggregate the CSV results into the tables of the paper.

    python scripts/summarize.py --modality tabular

For each method: metrics are averaged over instances within each seed, then
reported as mean ± standard deviation over seeds. Writes
results/<modality>/summary.csv and results/<modality>/summary.tex (table rows).
"""
import argparse
import glob
import json

import pandas as pd

from xaibench.utils import RESULTS_DIR

METRICS = ["fid_del", "fid_suf", "sparsity", "stability", "robustness", "consistency", "time_s"]
EXTRA = {"tabular": ["k", "size_abductive", "formally_sufficient", "sufficient_at_size_E",
                     "k_min_sufficient", "jaccard_abductive", "anchor_precision"],
         "timeseries": ["k", "mean_lag_topk"],
         "text": ["k", "n_words"],
         "image": ["k", "n_superpixels"],
         "graph": ["k", "d", "auc_gt", "auc_gt_signed", "precision_gt"]}

parser = argparse.ArgumentParser()
parser.add_argument("--modality", required=True)
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()

folder = RESULTS_DIR / args.modality / ("smoke" if args.smoke else "")
files = sorted(glob.glob(str(folder / "seed*.csv")))
if not files:
    raise SystemExit(f"No result files in {folder}")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
cols = [c for c in METRICS + EXTRA.get(args.modality, []) if c in df.columns]
df[cols] = df[cols].astype(float)

per_seed = df.groupby(["method", "seed"])[cols].mean()
summary = per_seed.groupby("method").agg(["mean", "std"])
n_inst = df.groupby(["method", "seed"]).size().groupby("method").mean()
if "top_variable" in df.columns:
    print("\nMost frequent variable in the top-k entries (share of instances):")
    print((df.groupby("method")["top_variable"].value_counts(normalize=True)
             .groupby(level=0).head(3).round(2)).to_string())
summary.to_csv(folder / "summary.csv")

print(f"\n{args.modality}: {len(files)} seed file(s), instances per seed: {n_inst.to_dict()}\n")
for f in sorted(glob.glob(str(folder / "model_seed*.json"))):
    m = json.load(open(f))
    print(f"  model seed {m['seed']}: accuracy {m['accuracy']:.3f}, balanced acc. "
          f"{m.get('balanced_accuracy', float('nan')):.3f}, AUC {m.get('auc', float('nan')):.3f}")

def fmt(mean, std):
    return f"{mean:.3f}" if pd.isna(std) else f"{mean:.3f} $\\pm$ {std:.3f}"

lines = []
print()
print("method".ljust(10) + "".join(c[:11].rjust(13) for c in cols))
for method, row in summary.iterrows():
    print(method.ljust(10) + "".join(f"{row[(c, 'mean')]:13.3f}" for c in cols))
    lines.append(f" & {method} & " + " & ".join(fmt(row[(c, 'mean')], row[(c, 'std')])
                                               for c in METRICS if c in cols) + r" \\")
with open(folder / "summary.tex", "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nWritten: {folder/'summary.csv'} and {folder/'summary.tex'}")

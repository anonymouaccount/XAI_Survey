import argparse
import importlib

MODALITIES = ["tabular", "timeseries", "text", "image", "graph"]

parser = argparse.ArgumentParser()
parser.add_argument("--modality", required=True, choices=MODALITIES)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--smoke", action="store_true", help="tiny run to test the pipeline")
args = parser.parse_args()

try:
    module = importlib.import_module(f"xaibench.{args.modality}")
except ModuleNotFoundError as e:
    if e.name == f"xaibench.{args.modality}":
        raise SystemExit(f"xaibench/{args.modality}.py is not written yet.")
    raise   # a package used inside the module is missing: show the real error
module.main(seed=args.seed, smoke=args.smoke)

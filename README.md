<<<<<<< HEAD
# From Theory to Evaluation: An Experimental Survey on XAI — benchmark code (v2)

Unified benchmark of XAI methods on image, text, tabular, graph and time-series data.
Every modality trains **one** model, explains the **same** test instances with every
method, and evaluates all explanations with the **same** metrics (`src/xaibench/metrics.py`),
which implement Section 5.3 of the paper.

```
xai-benchmark/
├── requirements/          binary.txt, pure.txt, torch.txt (exact package lists)
├── scripts/
│   ├── check_cluster.sh   run once on the cluster, tells which versions to download
│   ├── prepare_offline.sh run on your laptop (internet): downloads packages, data, weights
│   ├── setup_cluster.sh   run on the cluster (no internet): creates the environment
│   ├── check_env.py       prints package versions and GPU status
│   └── run.py             entry point: python scripts/run.py --modality tabular --seed 0
├── slurm/                 run_cpu.sbatch (tabular), run_gpu.sbatch (other modalities)
├── src/xaibench/
│   ├── metrics.py         Fid_del, Fid_suf, Gini sparsity, stability, robustness, consistency
│   ├── utils.py           seeds, paths, CSV results writer
│   └── <modality>.py      one file per modality (added step by step)
├── tests/test_metrics.py  checks the metrics on cases with a known answer
├── results/               CSV files used to build the tables of the paper
└── legacy_notebooks/      the original Colab notebooks (kept for the record, NOT used)
```

## Workflow (same method as CaHTGP)

The cluster has no internet, so everything is downloaded on the laptop, then the whole
project folder is copied to the cluster.

### 1. On the laptop: download packages, data and weights

Requirements: bash (Linux, macOS, WSL or Git Bash on Windows), `pip`, `curl`, and a Kaggle API
token in `~/.kaggle/kaggle.json` (Kaggle > Settings > Create New Token).

```bash
bash scripts/prepare_offline.sh
```
This fills `local_repo/` (wheels for Linux + Python 3.11 + PyTorch 2.11/CUDA 13.0, the same
PyTorch as CaHTGP), `data/raw/` and `models/`. Expected size: a few GB.

To download the datasets by hand instead, put these zips in `data/raw/`:
- https://www.kaggle.com/datasets/brandao/diabetes
- https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification
- https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
- https://www.kaggle.com/datasets/stackoverflow/stack-overflow-tag-network
- https://www.kaggle.com/datasets/sulphatet/daily-weather-data-40-years

### 2. Copy the project to the cluster

```bash
rsync -avP ./ r.abidi@frontal:/home/cril/r.abidi/xai/xai/
```
Later, to send only code changes:
```bash
rsync -avP --exclude local_repo --exclude env --exclude data --exclude models --exclude results --exclude logs \
      ./ r.abidi@frontal:/home/cril/r.abidi/xai/xai/
```
To bring the results back:
```bash
rsync -avP r.abidi@frontal:/home/cril/r.abidi/xai/xai/results/ ./results/
```

### 3. On the cluster: create the environment (once)

```bash
cd /home/cril/r.abidi/xai/xai
bash scripts/setup_cluster.sh
```
It creates `env/`, installs everything from `local_repo/` with `--no-index`, runs the tests
and prints the package versions. All tests must pass.

### 4. Run

```bash
source env/bin/activate
python scripts/run.py --modality tabular --seed 0 --smoke    # quick check (a few minutes)
mkdir -p logs
sbatch slurm/run_cpu.sbatch tabular                          # seeds 0-2, CPU
sbatch slurm/run_gpu.sbatch image                            # seeds 0-2, GPU (quad_rtx_8000)
squeue -u r.abidi
```
Submit from the same terminal/machine where `sbatch` worked for CaHTGP.

## Working in VS Code

- **Local editing**: open the folder, install the *Python* extension, and select a local
  virtual environment (a CPU-only install is enough to run the tests and `--smoke` runs).
- **Editing directly on the cluster**: the *Remote - SSH* extension works without internet on
  the cluster if you add this to your VS Code settings, so that the server is copied from
  your laptop instead of downloaded:
  ```json
  "remote.SSH.localServerDownload": "always"
  ```

## Tests

```bash
pytest -q
```
The tests check the metrics on a toy model that depends on a single feature, where the correct
answers are known (e.g. an explanation pointing to that feature has sufficiency 1; a random
explanation is unstable; the Gini sparsity of a set of m features out of d equals 1 - m/d).

## Protocol (summary of Table 4 of the paper)

| Modality | Model | Methods |
|---|---|---|
| Tabular (Diabetes 130-US) | gradient-boosted trees | PyXAI (abductive), Anchors, LIME, TreeSHAP, random |
| Time series (Daily Weather) | LSTM on windows, chronological split | SHAP, LIME, Integrated Gradients, random |
| Text (IMDB) | fine-tuned DistilBERT | LIME, DeepLIFT, random |
| Image (Cats vs Dogs) | fine-tuned ResNet-18 (has global average pooling, so CAM is valid) | CAM, Grad-CAM, LIME, SHAP, random |
| Graph (BA-Shapes, Stack Overflow, MUTAG) | 3-layer GCN | GNNExplainer, PGExplainer, GraphLIME, XGNN (MUTAG), random |

All metrics: 200 correctly classified test instances, 3 seeds, values in [0, 1], higher is better.
=======
# XAI_Survey
>>>>>>> f2f5414a0514919506d964aead0f41d5af238352

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
│   ├── prepare_offline.sh run on your laptop : downloads packages, data, weights
│   ├── setup_cluster.sh   run on the cluster : creates the environment
│   ├── check_env.py       prints package versions and GPU status
│   └── run.py             entry point: python scripts/run.py
├── slurm/                 run_cpu.sbatch (tabular), run_gpu.sbatch
├── src/xaibench/
│   ├── metrics.py         Fid_del, Fid_suf, Gini sparsity, stability, robustness, consistency
│   ├── utils.py           seeds, paths, CSV results writer
│   └── <modality>.py      one file per modality (added step by step)
├── tests/test_metrics.py  checks the metrics
├── results/               CSV files used to build the tables
└── legacy_notebooks/     
```

To download the datasets by hand instead, put these zips in `data/raw/`:
- https://www.kaggle.com/datasets/brandao/diabetes
- https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification
- https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
- https://www.kaggle.com/datasets/stackoverflow/stack-overflow-tag-network
- https://www.kaggle.com/datasets/sulphatet/daily-weather-data-40-years

## Protocol

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

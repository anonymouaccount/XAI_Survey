

from __future__ import annotations

import csv
import logging
import os
import platform
import random
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Root of the repository (…/xai-benchmark)
ROOT = Path(__file__).resolve().parents[2]

# Data and model folders can be moved (e.g. to a scratch disk on the cluster)
# with environment variables; defaults are inside the repository.
DATA_DIR = Path(os.environ.get("XAIB_DATA", ROOT / "data"))
MODELS_DIR = Path(os.environ.get("XAIB_MODELS", ROOT / "models"))
RESULTS_DIR = Path(os.environ.get("XAIB_RESULTS", ROOT / "results"))


def set_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return np.random.default_rng(seed)


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def run_metadata(modality: str, seed: int) -> Dict[str, Any]:
    return {
        "modality": modality,
        "seed": seed,
        "date": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "commit": git_commit(),
        "slurm_job": os.environ.get("SLURM_JOB_ID", ""),
    }


class ResultsWriter:

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            backup = self.path.with_name(self.path.name + ".bak")
            n = 0
            while backup.exists():
                n += 1
                backup = self.path.with_name(f"{self.path.name}.bak{n}")
            self.path.rename(backup)
            print(f"Previous results moved to {backup}")
        self._fields = None

    def write(self, row: Dict[str, Any]) -> None:
        new_file = not self.path.exists()
        if self._fields is None:
            if new_file:
                self._fields = list(row.keys())
            else:
                with open(self.path, newline="") as f:
                    self._fields = next(csv.reader(f))
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._fields, extrasaction="ignore")
            if new_file:
                w.writeheader()
            w.writerow(row)

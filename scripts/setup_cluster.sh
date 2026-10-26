#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3.11}"

$PY -m venv env
source env/bin/activate
pip install --no-index --find-links local_repo setuptools wheel
pip install --no-index --find-links local_repo -r requirements/torch.txt
pip install --no-index --find-links local_repo -r requirements/binary.txt
pip install --no-index --find-links local_repo --no-deps --no-build-isolation pyxai docplex lime anchor-exp captum
pip install --no-index --no-deps --no-build-isolation -e .

python -m pytest -q
python scripts/check_env.py
echo "Environment ready. Activate it with:  source env/bin/activate"

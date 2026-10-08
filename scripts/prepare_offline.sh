#!/usr/bin/env bash
# =============================================================================
# Run on your LAPTOP (with internet): Linux, macOS, or WSL / Git Bash on Windows.
# Same method as for CaHTGP: everything the cluster needs is downloaded here,
# then the whole project folder is copied to the cluster.
#
#   local_repo/   Python packages (wheels) for the cluster: Linux, Python 3.11
#   data/raw/     the datasets
#   models/       pretrained weights (ResNet-18, DistilBERT)
#
# Values below come from scripts/check_cluster.sh on frontal
# (Python 3.11, glibc 2.39, PyTorch 2.11 + CUDA 13.0 already working on quad_rtx_8000).
PYVER="${PYVER:-3.11}"
CUDA_TAG="${CUDA_TAG:-cu130}"
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p local_repo data/raw models/torch/hub/checkpoints models/hf

PYNODOT=${PYVER/./}
# cluster glibc is 2.39: accept every manylinux tag from 2.17 to 2.39
PL=""; for v in $(seq 39 -1 17); do PL="$PL --platform manylinux_2_${v}_x86_64"; done
TARGET="--python-version $PYVER --implementation cp --abi cp$PYNODOT \
  $PL --platform manylinux2014_x86_64 \
  --platform manylinux2010_x86_64 --platform manylinux1_x86_64 --platform any"

echo "== 1/4 PyTorch 2.11 ($CUDA_TAG) =="
pip download -d local_repo --only-binary=:all: $TARGET \
  --index-url https://download.pytorch.org/whl/$CUDA_TAG --extra-index-url https://pypi.org/simple \
  -r requirements/torch.txt

echo "== 2/4 Other packages =="
pip download -d local_repo --only-binary=:all: $TARGET -r requirements/binary.txt
grep -v '^pyxai' requirements/pure.txt > /tmp/pure_nopyxai.txt
pip download -d local_repo --no-deps $TARGET -r /tmp/pure_nopyxai.txt
pip download -d local_repo --no-deps --only-binary=:all: $TARGET pyxai==2.0.1

echo "== 3/4 Pretrained weights =="
curl -L -o models/torch/hub/checkpoints/resnet18-f37072fd.pth \
  https://download.pytorch.org/models/resnet18-f37072fd.pth
pip install -q "huggingface_hub>=0.23"
python -c "from huggingface_hub import snapshot_download; snapshot_download('distilbert-base-uncased', local_dir='models/hf/distilbert-base-uncased', allow_patterns=['*.json','*.txt','*.safetensors'])"

echo "== 4/4 Datasets =="
# Needs a Kaggle API token in ~/.kaggle/kaggle.json (Kaggle > Settings > Create New Token).
# Alternatively, download the five zips by hand (links in README.md) into data/raw/.
pip install -q kaggle
for ds in brandao/diabetes \
          samuelcortinhas/cats-and-dogs-image-classification \
          lakshmi25npathi/imdb-dataset-of-50k-movie-reviews \
          stackoverflow/stack-overflow-tag-network \
          sulphatet/daily-weather-data-40-years; do
  kaggle datasets download -d "$ds" -p data/raw
done
curl -L -o data/raw/MUTAG.zip https://www.chrsmrrs.de/graphkerneldatasets/MUTAG.zip

du -sh local_repo data models
echo "Done. Copy the whole project to the cluster, e.g.:"
echo "  rsync -avP ./ r.abidi@frontal:/home/cril/r.abidi/xai/xai/"

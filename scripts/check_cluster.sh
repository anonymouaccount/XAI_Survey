#!/usr/bin/env bash
# Run this ONCE on the cluster login node and send me the output.
# It tells us which Python, glibc and CUDA versions to download packages for.
echo "== Python versions available =="
for p in python3 python3.10 python3.11 python3.12; do command -v $p >/dev/null && echo "$p -> $($p --version 2>&1)"; done
echo; echo "== Environment modules (python / cuda / conda) =="
(module avail 2>&1 | grep -iE "python|cuda|conda|anaconda|miniforge" | head -30) || echo "no module command"
echo; echo "== glibc version =="
ldd --version | head -1
echo; echo "== GPU driver (run on a GPU node if the login node has none) =="
(nvidia-smi --query-gpu=name,driver_version --format=csv 2>/dev/null && nvidia-smi | grep "CUDA Version") || echo "nvidia-smi not available here"
echo; echo "== SLURM partitions =="
sinfo -o "%P %G %l %c %m" 2>/dev/null | head -20
echo; echo "== Disk quotas (home / scratch) =="
(quota -s 2>/dev/null || df -h $HOME) | head -10

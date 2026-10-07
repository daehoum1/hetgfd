#!/usr/bin/env bash
# One worker per GPU, all sharing one work queue (resumable: re-run the same command after an interruption).
#   usage: scripts/launch.sh <run_hgt.py|run_hgnnac.py> <workdir> [script options...]
#   e.g.   scripts/launch.sh run_hgt.py runs/nc_text --tasks nc --hgt text --methods zero mean knn svd fp pcfi hetgfd
#          GPUS="0 1" scripts/launch.sh run_hgnnac.py runs/hgnnac
set -e
cd "$(dirname "$0")/.."
SCRIPT=$1; WORK=$2; shift 2
GPUS=${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')}
LAUNCH=L$(date +%Y%m%d_%H%M%S)
mkdir -p "$WORK/logs"
for g in $GPUS; do
  CUDA_VISIBLE_DEVICES=$g nohup python "scripts/$SCRIPT" --workdir "$WORK" --worker "gpu$g" --launch "$LAUNCH" "$@" \
    >> "$WORK/logs/gpu$g.out" 2>&1 &
  echo "gpu$g: pid $!"
done
echo "logs: $WORK/logs/  results: $WORK/results/  (summary_*.csv is rewritten after every run)"

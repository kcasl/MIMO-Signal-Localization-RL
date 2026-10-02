#!/usr/bin/env bash
set -euo pipefail
cd /data/jw_workspace/MIMO/MIMOPathFinder
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate MIMO
export CUDA_VISIBLE_DEVICES=0
exec python -u train_adwa_curriculum.py --timesteps 2000000 \
  2>&1 | tee /data/jw_workspace/MIMO/MIMOPathFinder/outputs/train_adwa_curriculum.log

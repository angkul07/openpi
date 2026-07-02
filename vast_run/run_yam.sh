#!/bin/bash
# YAM pi0-fast training pipeline on 2x A100-80GB. Run inside tmux.
exec > >(tee -a /workspace/train.log) 2>&1   # log to file AND tmux pane
cd /workspace/openpi

export HF_HOME=/workspace/.hf_home
export HF_LEROBOT_HOME=/workspace/.hf_home/lerobot
export OPENPI_DATA_HOME=/workspace/.openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=<REDACTED_WANDB_KEY>
export WANDB_ENTITY=kavinrajkr60-dsfsd
export WANDB_MODE=online

CONFIG=pi0_fast_yam_low_mem_finetune
NORM_DIR="assets/${CONFIG}/Kavin60606/yam_pi0fast_train"

echo "===== [1/2] norm stats (downloads dataset; retries HF 429) $(date -u) ====="
if [ -d "$NORM_DIR" ] && [ -n "$(ls -A "$NORM_DIR" 2>/dev/null)" ]; then
  echo "norm stats already present at $NORM_DIR -- skipping"
else
  for i in $(seq 1 40); do
    echo "--- norm-stats attempt $i $(date -u +%H:%M:%S) ---"
    uv run scripts/compute_norm_stats.py --config-name "$CONFIG" && break
    echo "[norm] attempt $i failed (likely HF 429); sleeping 330s then resuming..."
    sleep 330
  done
fi

if [ ! -d "$NORM_DIR" ] || [ -z "$(ls -A "$NORM_DIR" 2>/dev/null)" ]; then
  echo "ERROR: norm stats not computed after retries; aborting before training."
  exit 1
fi

echo "===== [2/2] train (2x A100 data-parallel) $(date -u) ====="
uv run scripts/train.py "$CONFIG" \
  --exp-name yam_run --fsdp-devices 1 --overwrite \
  --project-name pi0-fast-modal

echo "===== PIPELINE FINISHED (exit $?) $(date -u) ====="

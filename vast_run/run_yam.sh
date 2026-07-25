#!/bin/bash
# YAM pi0-fast teleop-oversampling co-fine-tune on 2x A100-80GB. Run inside tmux.
#
#   ./vast_run/run_yam.sh                        # E-A  (p_teleop=0.50, 50k steps)
#   ./vast_run/run_yam.sh pi0_fast_yam_mix_eb    # E-B  (p_teleop=0.625, 50k steps)
#   ./vast_run/run_yam.sh pi0_fast_yam_mix_ec    # E-C  (p_teleop=0.50, 100k steps)
#
# Prereqs on the box:
#   uv run vast_run/dl.py                        # both datasets -> $HF_LEROBOT_HOME
#   uv run vast_run/preflight_checks.py          # BLOCKING gripper-convention check
#   export WANDB_API_KEY=...                     # or put it in vast_run/env.local
set -o pipefail
cd /workspace/openpi || exit 1

CONFIG="${1:-pi0_fast_yam_mix_ea}"
EXP_NAME="${2:-${CONFIG#pi0_fast_yam_mix_}}"
LOG=/workspace/train_${CONFIG}.log
exec > >(tee -a "$LOG") 2>&1   # log to file AND tmux pane

export HF_HOME=/workspace/.hf_home
export HF_LEROBOT_HOME=/workspace/.hf_home/lerobot
export OPENPI_DATA_HOME=/workspace/.openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export WANDB_ENTITY="${WANDB_ENTITY:-kavinrajkr60-dsfsd}"
export WANDB_MODE="${WANDB_MODE:-online}"
# Secrets live outside git. Put `export WANDB_API_KEY=...` in vast_run/env.local
# (untracked) or export it before launching.
[ -f vast_run/env.local ] && . vast_run/env.local
: "${WANDB_API_KEY:?set WANDB_API_KEY (env or vast_run/env.local) before launching}"

# Frames drawn through the training sampler to estimate QUANTILES norm stats.
# 200k is far more than quantile estimation needs and takes minutes with
# --skip-videos (norm stats never touch pixels).
NORM_FRAMES="${NORM_FRAMES:-200000}"

echo "===== [0/3] resolve config $(date -u) ====="
# Fail fast on a bad config / missing dataset before burning GPU hours, and print
# the actual sampled mixture so the ratio in the log is the ratio that ran.
ASSET_ID_FILE=/tmp/${CONFIG}_asset_id.txt
uv run python - "$CONFIG" "$ASSET_ID_FILE" <<'PY' || exit 1
import pathlib
import sys
import openpi.training.config as _config

cfg = _config.get_config(sys.argv[1])
data = cfg.data.create(cfg.assets_dirs, cfg.model)
batch = cfg.batch_size
print(f"config          : {cfg.name}")
print(f"steps           : {cfg.num_train_steps}  (warmup {cfg.lr_schedule.warmup_steps}, "
      f"decay {cfg.lr_schedule.decay_steps}, peak {cfg.lr_schedule.peak_lr})")
print(f"batch / workers : {batch} / {cfg.num_workers}    save_interval {cfg.save_interval}")
print(f"asset_id        : {data.asset_id}")
assert data.mixture, "config has no mixture -- wrong config name?"
assert cfg.lr_schedule.decay_steps == cfg.num_train_steps, "decay_steps must equal num_train_steps"
total = sum(s.samples_per_batch for s in data.mixture)
assert total == batch, f"mixture counts {total} != batch size {batch}"
for s in data.mixture:
    p = s.samples_per_batch / batch
    epochs = p * batch * cfg.num_train_steps
    print(f"  {s.repo_id:52s} {s.samples_per_batch:3d}/{batch} per batch "
          f"({p:.1%} gradient share), holdout={s.holdout_fraction}, "
          f"{epochs:,.0f} frame-visits over the run")
pathlib.Path(sys.argv[2]).write_text(data.asset_id)
print("CONFIG OK")
PY

NORM_DIR="assets/${CONFIG}/$(cat "$ASSET_ID_FILE")"
echo "norm stats dir  : $NORM_DIR"

echo "===== [1/3] norm stats over the SAMPLED mixture $(date -u) ====="
# Computed through the same fixed-ratio sampler as training: raw-storage stats
# would be ego-dominated (33/67) and would misnormalize the teleop grippers.
if [ -s "$NORM_DIR/norm_stats.json" ]; then
  echo "norm stats already present at $NORM_DIR -- skipping"
else
  for i in $(seq 1 40); do
    echo "--- norm-stats attempt $i $(date -u +%H:%M:%S) ---"
    uv run scripts/compute_norm_stats.py --config-name "$CONFIG" \
      --max-frames "$NORM_FRAMES" --skip-videos && break
    echo "[norm] attempt $i failed (likely HF 429); sleeping 330s then resuming..."
    sleep 330
  done
fi

if [ ! -s "$NORM_DIR/norm_stats.json" ]; then
  echo "ERROR: norm stats not computed after retries; aborting before training."
  exit 1
fi

echo "===== [2/3] train (2x A100 data-parallel) $(date -u) ====="
# NOTE: fresh run from pi0_fast_base, NOT a resume of the 7k checkpoint -- a resume
# would drag along its exhausted cosine schedule and the old 50/50 norm stats.
uv run scripts/train.py "$CONFIG" \
  --exp-name "$EXP_NAME" --fsdp-devices 1 --overwrite \
  --project-name pi0-fast-modal

echo "===== [3/3] PIPELINE FINISHED (exit $?) $(date -u) ====="

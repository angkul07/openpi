#!/bin/bash
# YAM pi0-fast teleop-oversampling co-fine-tune on 2x A100-80GB. Run inside tmux.
#
# 15h mixture (full datasets from $HF_LEROBOT_HOME):
#   ./vast_run/run_yam.sh                        # E-A  (p_teleop=0.50, 50k steps)
#   ./vast_run/run_yam.sh pi0_fast_yam_mix_eb    # E-B  (p_teleop=0.625, 50k steps)
#   ./vast_run/run_yam.sh pi0_fast_yam_mix_ec    # E-C  (p_teleop=0.50, 100k steps)
#
# 7h mixture (pre-selected local subsets under /workspace/data/yam7h):
#   ./vast_run/run_yam.sh pi0_fast_yam7h_ea      # E-A  (p_teleop=0.50, 23.6k steps)
#   ./vast_run/run_yam.sh pi0_fast_yam7h_eb      # E-B  (p_teleop=0.625, 23.6k steps)
#   ./vast_run/run_yam.sh pi0_fast_yam7h_ec      # E-C  (p_teleop=0.50, 47.2k steps)
#
# Prereqs on the box:
#   uv run vast_run/dl.py                        # 15h arms only: datasets -> $HF_LEROBOT_HOME
#                                                # 7h arms ship their own data; nothing to download
#   uv run vast_run/preflight_checks.py          # BLOCKING gripper-convention check
#   export WANDB_API_KEY=...                     # or put it in vast_run/env.local
set -o pipefail
cd /workspace/openpi || exit 1

CONFIG="${1:-pi0_fast_yam_mix_ea}"
# Strip whichever family prefix matches so EXP_NAME is just the arm ("ea"), keeping
# checkpoints at checkpoints/<config>/<arm>/ for both families.
_ARM="${CONFIG#pi0_fast_yam_mix_}"
_ARM="${_ARM#pi0_fast_yam7h_}"
EXP_NAME="${2:-$_ARM}"

# Logs, all under /workspace/logs/<config>/:
#   pipeline.log     everything this script prints (also on the tmux pane)
#   norm_stats.log   stage 1 only
#   train.log        stage 2 stdout (tqdm + the log_interval averages)
# Plus checkpoints/<config>/<exp>/train_metrics.log, written by train.py itself:
# one line per training step, un-averaged.
LOG_DIR=/workspace/logs/${CONFIG}
mkdir -p "$LOG_DIR"
NORM_LOG="$LOG_DIR/norm_stats.log"
TRAIN_LOG="$LOG_DIR/train.log"
exec > >(tee -a "$LOG_DIR/pipeline.log") 2>&1   # log to file AND tmux pane

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
import json
import os
import pathlib
import sys
import lerobot.common.datasets.lerobot_dataset as _lerobot
import openpi.training.config as _config

cfg = _config.get_config(sys.argv[1])
data = cfg.data.create(cfg.assets_dirs, cfg.model)
batch = cfg.batch_size
print(f"config          : {cfg.name}")
print(f"steps           : {cfg.num_train_steps}  (warmup {cfg.lr_schedule.warmup_steps}, "
      f"decay {cfg.lr_schedule.decay_steps}, peak {cfg.lr_schedule.peak_lr})")
print(f"batch / workers : {batch} / {cfg.num_workers}")
print(f"checkpoints     : every {cfg.save_interval} steps, keep {cfg.max_to_keep} most recent "
      f"(keep_period={cfg.keep_period})")
print(f"asset_id        : {data.asset_id}")
assert data.mixture, "config has no mixture -- wrong config name?"
assert cfg.lr_schedule.decay_steps == cfg.num_train_steps, "decay_steps must equal num_train_steps"
total = sum(s.samples_per_batch for s in data.mixture)
assert total == batch, f"mixture counts {total} != batch size {batch}"
for s in data.mixture:
    p = s.samples_per_batch / batch
    epochs = p * batch * cfg.num_train_steps
    held = f"{len(s.exclude_episodes)} episodes" if s.exclude_episodes else f"fraction {s.holdout_fraction}"
    print(f"  {s.repo_id:52s} {s.samples_per_batch:3d}/{batch} per batch "
          f"({p:.1%} gradient share), held out: {held}, "
          f"{epochs:,.0f} frame-visits over the run")

# Both datasets are authored as LeRobot v2.1, so nothing may be converted or rewritten:
# training reads the downloaded files as-is. Assert that here rather than trusting it --
# a version mismatch is exactly what would make lerobot reach for a migration path.
lerobot_home = pathlib.Path(os.environ.get("HF_LEROBOT_HOME", pathlib.Path.home() / ".cache/huggingface/lerobot"))
for s in data.mixture:
    root = pathlib.Path(s.root) if s.root else lerobot_home / s.repo_id
    info = root / "meta" / "info.json"
    if not info.is_file():
        raise SystemExit(f"ERROR: {s.repo_id} not found at {root} -- download it before training (vast_run/dl.py)")
    version = json.loads(info.read_text())["codebase_version"]
    if version != _lerobot.CODEBASE_VERSION:
        raise SystemExit(
            f"ERROR: {s.repo_id} is {version}, but this lerobot expects {_lerobot.CODEBASE_VERSION}. "
            "Refusing to run: converting/migrating the dataset in place is not part of this pipeline."
        )
    print(f"  {s.repo_id:52s} {version} at {root} (read as-is, no conversion)")

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
    echo "--- norm-stats attempt $i $(date -u +%H:%M:%S) ---" | tee -a "$NORM_LOG"
    uv run scripts/compute_norm_stats.py --config-name "$CONFIG" \
      --max-frames "$NORM_FRAMES" --skip-videos 2>&1 | tee -a "$NORM_LOG" && break
    echo "[norm] attempt $i failed (likely HF 429); sleeping 330s then resuming..."
    sleep 330
  done
fi
echo "norm stats log  : $NORM_LOG"

if [ ! -s "$NORM_DIR/norm_stats.json" ]; then
  echo "ERROR: norm stats not computed after retries; aborting before training."
  exit 1
fi

echo "===== [2/3] train (2x A100 data-parallel) $(date -u) ====="
# Re-running this script must not destroy a run in progress: if the checkpoint dir
# already has a numbered checkpoint, continue it; otherwise start clean. FRESH=1
# forces a restart (wipes the checkpoint dir).
# NOTE: "fresh" means from pi0_fast_base, NOT from the old 7k checkpoint -- that
# would drag along its exhausted cosine schedule and the old 50/50 norm stats.
CKPT_DIR="checkpoints/${CONFIG}/${EXP_NAME}"
shopt -s nullglob
EXISTING=("$CKPT_DIR"/[0-9]*)
shopt -u nullglob
if [ "${FRESH:-0}" = "1" ]; then
  MODE=--overwrite
  echo "mode            : FRESH (wiping $CKPT_DIR)"
elif [ ${#EXISTING[@]} -gt 0 ]; then
  MODE=--resume
  echo "mode            : RESUME (existing checkpoints: ${EXISTING[*]##*/})"
else
  MODE=--overwrite
  echo "mode            : fresh start (no checkpoints in $CKPT_DIR)"
fi
echo "train log       : $TRAIN_LOG"
echo "per-step metrics: $CKPT_DIR/train_metrics.log"

# --fsdp-devices 1 = pure data parallel: mesh is (num_gpus, 1), so batch 64 splits
# 32/GPU across the 2 A100s and every GPU holds a full copy of the LoRA model.
uv run scripts/train.py "$CONFIG" \
  --exp-name "$EXP_NAME" --fsdp-devices 1 "$MODE" \
  --project-name pi0-fast-modal 2>&1 | tee -a "$TRAIN_LOG"

echo "===== [3/3] PIPELINE FINISHED (exit $?) $(date -u) ====="

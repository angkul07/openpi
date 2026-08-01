#!/bin/bash
# YAM teleop-oversampling co-fine-tune on 2x A100-80GB. Run inside tmux.
# Handles both the pi0-FAST and the pi0.5 config families.
#
# 15h mixture (full datasets from $HF_LEROBOT_HOME):
#   ./vast_run/run_yam.sh                        # E-A  (p_teleop=0.50, 50k steps)
#   ./vast_run/run_yam.sh pi0_fast_yam_mix_eb    # E-B  (p_teleop=0.625, 50k steps)
#   ./vast_run/run_yam.sh pi0_fast_yam_mix_ec    # E-C  (p_teleop=0.50, 100k steps)
#
# 7h mixture, pi0-FAST (pre-selected local subsets under /workspace/data/yam7h):
#   ./vast_run/run_yam.sh pi0_fast_yam7h_ea      # E-A  (p_teleop=0.50, 23.6k steps)
#   ./vast_run/run_yam.sh pi0_fast_yam7h_eb      # E-B  (p_teleop=0.625, 23.6k steps)
#   ./vast_run/run_yam.sh pi0_fast_yam7h_ec      # E-C  (p_teleop=0.50, 47.2k steps)
#
# 7h mixture, pi0.5 (same data, same schedule, flow-matching head):
#   ./vast_run/run_yam.sh pi05_yam7h_ea          # E-A  (p_teleop=0.50, 23.6k steps)
#   ./vast_run/run_yam.sh pi05_yam7h_eb          # E-B  (p_teleop=0.625, 23.6k steps)
#   ./vast_run/run_yam.sh pi05_yam7h_ec          # E-C  (p_teleop=0.50, 47.2k steps)
#
# 7h PRE-MERGED 50/50, pi0.5 -- the openpi counterpart of the LeRobot run:
#   ./vast_run/run_yam.sh pi05_50run_ea          # 50/50 by storage, 17.7k steps
#
#   One dataset, not two roots: the 50/50 is baked into storage, so the sampler draws
#   all 64 from it. Fetch it first (~2.5 GB, private repo -- needs `hf auth login`):
#     hf download angkul07/50_run_v21_fixed 50_run_v21.tar \
#        --repo-type dataset --local-dir /workspace
#     tar xf /workspace/50_run_v21.tar -C /workspace
#     mv /workspace/50_run_old /workspace/50_run      # tar unpacks under its build name
#   Point YAM50RUN_ROOT elsewhere if you keep it off /workspace/50_run.
#   It is LeRobot v2.1 on purpose -- stage [0] refuses anything this lerobot rev does
#   not expect, and will NOT migrate a v3.0 tree in place.
#
# Prereqs on the box:
#   uv run vast_run/dl.py                        # 15h arms only: datasets -> $HF_LEROBOT_HOME
#                                                # 7h arms ship their own data; nothing to download
#                                                # pi05_50run_ea: see the tar fetch above
#   uv run vast_run/preflight_checks.py          # BLOCKING gripper-convention check
#   uv run vast_run/pi05/pi05_preflight.py pi05_yam7h_ea   # pi0.5 arms only, see below
#   export WANDB_API_KEY=...                     # or put it in vast_run/env.local
#
# pi05_yam7h_* arms: norm stats are IDENTICAL to the matching pi0-FAST arm (quantile
# stats are computed after data_transforms and before model_transforms, and
# use_quantile_norm is True for both PI0_FAST and PI05). Copy them across and stage
# [1/3] is a no-op:
#   mkdir -p assets/pi05_yam7h_ea
#   cp -r assets/pi0_fast_yam7h_ea/yam7h_p50 assets/pi05_yam7h_ea/
#
# This does NOT apply to pi05_abcego_sd or pi05_50run_ea: neither has a pi0-FAST
# counterpart, and both draw a different distribution, so their stats must actually be
# computed. Expect stage [1/3] to run for ~10 min on those. Copying yam7h_* stats onto
# them would silently normalise against the wrong distribution.
set -o pipefail
cd /workspace/openpi || exit 1

CONFIG="${1:-pi0_fast_yam_mix_ea}"
# Strip whichever family prefix matches so EXP_NAME is just the arm ("ea"), keeping
# checkpoints at checkpoints/<config>/<arm>/ for every family.
_ARM="${CONFIG#pi0_fast_yam_mix_}"
_ARM="${_ARM#pi0_fast_yam7h_}"
_ARM="${_ARM#pi05_yam7h_}"
_ARM="${_ARM#pi05_abcego_}"
_ARM="${_ARM#pi05_50run_}"
EXP_NAME="${2:-$_ARM}"

# W&B project, per family, so the pi0-FAST and pi0.5 runs do not land in one soup.
# Override with WANDB_PROJECT=... if you want them side by side.
case "$CONFIG" in
  pi05_*) PROJECT="${WANDB_PROJECT:-pi05-yam}" ;;
  *)      PROJECT="${WANDB_PROJECT:-pi0-fast-modal}" ;;
esac

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
print(f"model           : {type(cfg.model).__name__} type={cfg.model.model_type.name} "
      f"action_dim={cfg.model.action_dim} horizon={cfg.model.action_horizon} "
      f"max_token_len={cfg.model.max_token_len}")
print(f"steps           : {cfg.num_train_steps}  (warmup {cfg.lr_schedule.warmup_steps}, "
      f"decay {cfg.lr_schedule.decay_steps}, peak {cfg.lr_schedule.peak_lr})")
print(f"batch / workers : {batch} / {cfg.num_workers}")
print(f"checkpoints     : every {cfg.save_interval} steps, keep {cfg.max_to_keep} most recent "
      f"(keep_period={cfg.keep_period})")
print(f"asset_id        : {data.asset_id}")
print(f"quantile norm   : {data.use_quantile_norm}")
assert data.mixture, "config has no mixture -- wrong config name?"
assert cfg.lr_schedule.decay_steps == cfg.num_train_steps, "decay_steps must equal num_train_steps"
total = sum(s.samples_per_batch for s in data.mixture)
assert total == batch, f"mixture counts {total} != batch size {batch}"

# pi0.5 guard rails. These are the two failures that surface late and confusingly:
# action_dim != 32 blows up inside jit (weight_loaders._merge_params never checks
# shapes), and discrete_state_input=False silently removes proprioception entirely.
if cfg.model.model_type.name == "PI05":
    assert cfg.model.action_dim == 32, (
        f"pi0.5 needs action_dim=32 to match pi05_base's action_in/out_proj, got {cfg.model.action_dim}"
    )
    assert getattr(cfg.model, "discrete_state_input", True), (
        "discrete_state_input=False with pi05=True drops the state entirely -- "
        "embed_suffix emits no state token either. Leave it unset."
    )

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
#
# For a pi05_yam7h_* arm this stage should print "already present" -- the stats are
# byte-identical to the matching pi0_fast arm and you copied them in. If it starts
# computing, you skipped the copy; that is not fatal, just 10 wasted minutes.
# For pi05_abcego_sd / pi05_50run_ea there is nothing to copy: computing here is the
# CORRECT behaviour, not a sign you missed a step.
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
# NOTE: "fresh" means from the config's own base checkpoint (pi0_fast_base or
# pi05_base), NOT from any earlier run -- a resume across configs would drag along
# an exhausted cosine schedule and stale norm stats.
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
echo "wandb project   : $PROJECT"

# --fsdp-devices 1 = pure data parallel: mesh is (num_gpus, 1), so batch 64 splits
# 32/GPU across the 2 A100s and every GPU holds a full copy of the model. For pi0.5
# that copy also carries the 311M full-rank action expert and its AdamW state
# (~+4 GB/GPU vs the pi0-FAST arms).
uv run scripts/train.py "$CONFIG" \
  --exp-name "$EXP_NAME" --fsdp-devices 1 "$MODE" \
  --project-name "$PROJECT" 2>&1 | tee -a "$TRAIN_LOG"

echo "===== [3/3] PIPELINE FINISHED (exit $?) $(date -u) ====="

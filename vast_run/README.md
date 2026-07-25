# vast.ai run book — YAM pi0-FAST teleop-oversampling co-fine-tune

Two LeRobot v2.1 datasets, stored at **30% teleop / 70% ego** by frames after the
holdout is carved out, sampled at a **fixed per-batch ratio** so teleop's gradient
share is set explicitly and independently of how much of it is on disk:

| source | repo | train frames | hours | per batch (E-A) |
| --- | --- | --- | --- | --- |
| teleop (real robot) | `angkul07/abc-teleop` | 460,647 | 4.265 | 32 / 64 |
| ego (retargeted) | `angkul07/EgoDex-PickPlace-YAM-14dof-multiview` | 1,074,893 | 9.953 | 32 / 64 |
| *teleop holdout (not trained on)* | `/workspace/abc-teleop-holdout` | *77,806* | *0.720* | — |

Both datasets are already v2.1 — **nothing is converted**. `LeRobotDataset` reads
them from `$HF_LEROBOT_HOME/<repo_id>` as-is; openpi's `examples/*/convert_*.py`
scripts are manual-only and are never invoked by this pipeline. Stage [0/3] asserts
each source's `meta/info.json` `codebase_version` matches the pinned lerobot
(`v2.1`) and aborts otherwise, since a version mismatch is exactly what would send
lerobot looking for a migration path.

## Configs

| config | p_teleop | batch split T/E | steps | warmup |
| --- | --- | --- | --- | --- |
| `pi0_fast_yam_mix_ea` | 0.50 | 32 / 32 | 50,000 | 1,000 |
| `pi0_fast_yam_mix_eb` | 0.625 | 40 / 24 | 50,000 | 1,000 |
| `pi0_fast_yam_mix_ec` | 0.50 | 32 / 32 | 100,000 | 2,000 |

Everything else matches the previous 7k run: LoRA `gemma_2b_lora`, batch 64,
peak LR 3.5e-5 → 3.5e-6 cosine (decay_steps == num_train_steps), bf16, EMA off,
QUANTILES norm, delta joints + absolute grippers.

## Multi-GPU and resume

**2× A100-80GB, data parallel.** `--fsdp-devices 1` makes the mesh `(num_gpus, 1)`,
so batch 64 splits 32/GPU and each GPU holds a full copy of the LoRA model — same
setup as the previous run. `train.py` refuses to start if `batch_size` isn't
divisible by the device count. Raise `--fsdp-devices` only if a single GPU can't
hold the model; it shards parameters instead and costs communication.

**Resume.** Re-running `run_yam.sh` continues an interrupted run automatically:
it uses `--resume` when the checkpoint dir already has checkpoints and
`--overwrite` when it doesn't. `FRESH=1 ./vast_run/run_yam.sh <config>` forces a
restart from `pi0_fast_base`. The checkpoint carries step, params, optimizer state
and LR-schedule position; W&B reattaches to the same run via `wandb_id.txt`.

The data stream also resumes: the mixture sampler seeks to `latest_step + 1` in
O(1), so a resumed run draws the batches the interrupted one would have drawn
instead of replaying the stream from the beginning. (Stock openpi restarts the
shuffle on every resume — this only works for mixture configs.)

## Checkpoints and logs

All three arms save **every 1,000 steps and keep only the 4 most recent**
(`save_interval=1000`, `max_to_keep=4`, `keep_period=None` — nothing is pinned
permanently). That is a ~4,000-step window: eval or upload a checkpoint before it
rolls off, or bump `max_to_keep`.

| file | what |
| --- | --- |
| `/workspace/logs/<config>/pipeline.log` | everything `run_yam.sh` prints |
| `/workspace/logs/<config>/norm_stats.log` | stage 1 only |
| `/workspace/logs/<config>/train.log` | training stdout (tqdm + `log_interval` averages) |
| `checkpoints/<config>/<exp>/train_metrics.log` | **one line per training step**, un-averaged |

`train_metrics.log` is written by `train.py` from the same host transfer that
already happens every `log_interval` steps, so per-step fidelity costs no extra
device sync. Format: `step=N loss=... learning_rate=... grad_norm=... param_norm=...`

## Order of operations

```bash
export HF_HOME=/workspace/.hf_home
export HF_LEROBOT_HOME=/workspace/.hf_home/lerobot
echo 'export WANDB_API_KEY=...' > vast_run/env.local   # untracked; do not commit

# 1. data (~15h of video; resumable, waits out HF 429s)
uv run vast_run/dl.py

# 2. BLOCKING pre-flight: gripper convention + delta distributions (parquet only)
uv run vast_run/preflight_checks.py
#    -> eyeball /workspace/preflight/gripper_convention.png before continuing

# 3. norm stats over the sampled mixture + train (one script, inside tmux)
tmux new -s train './vast_run/run_yam.sh pi0_fast_yam_mix_ea'

# 4. push checkpoints
uv run vast_run/upload_ckpt.py --config pi0_fast_yam_mix_ea
```

## Things that will bite you

- **Norm stats are per-ratio and must be computed through the sampler.** Stats over
  raw storage would be ego-dominated (33/67) and would misnormalize the teleop
  grippers this experiment prioritizes. `run_yam.sh` does this; if you run it by
  hand: `uv run scripts/compute_norm_stats.py --config-name <cfg> --max-frames
  200000 --skip-videos`. `--skip-videos` is safe — norm stats only read
  state/actions from parquet — and turns hours into minutes.
  E-A and E-C share a ratio (`asset_id=yam_mix_p50`), so you can copy
  `assets/pi0_fast_yam_mix_ea/yam_mix_p50/` to
  `assets/pi0_fast_yam_mix_ec/yam_mix_p50/` instead of recomputing.
- **These are fresh runs from `pi0_fast_base`, not resumes of the 7k checkpoint.**
  Resuming would drag along the exhausted cosine schedule and the old 50/50
  EgoDex norm stats.
- **The 30/70 storage split does NOT change the training mixture.** The sampler draws
  a fixed 32 teleop + 32 ego per batch, so teleop's gradient share is exactly 50%
  whether it is 33% or 30% of what is on disk. Carving out the holdout changes *which*
  frames exist to be drawn, not their weight. Do not expect the loss curve to move
  because of it.
- **Teleop holdout: 249 episodes / 77,806 frames / 0.720 h**, generated by
  `vast_run/make_holdout.py` and copied to `/workspace/abc-teleop-holdout` as a
  standalone v2.1 dataset. Selection is by *duration* (episode lengths vary, so
  dropping 14.4% of episodes would not drop 14.4% of the hours) and lands training
  storage at 30.00%. Nothing was deleted from the source: the episodes are still on
  disk, and training simply never samples them via
  `exclude_episodes=_TELEOP_HOLDOUT_EPISODES` (`src/openpi/training/teleop_holdout.json`,
  checked in so the split travels with the code). The count is printed at startup.
  Inside the holdout folder episodes are renumbered `0..248` so it loads standalone;
  `holdout_episodes.json` keeps the original→local map. Ego gets no holdout —
  headline metrics are teleop-only by design.
  To regenerate at a different ratio: `uv run vast_run/make_holdout.py --dry-run
  --target-frac 0.25` (then re-copy the manifest and recompute norm stats).
- **Augmentation is training-only** (`DataConfig.train_only_transforms`), so it is
  off at eval/serving by construction: photometric jitter on all 3 cameras,
  random crop-and-resize on the **top camera only** (wrist views are pose-coupled),
  no flips/rotations, no noise on state or actions.
- `run_yam.sh` refuses to start if `decay_steps != num_train_steps` or if the
  per-batch counts don't sum to the batch size.

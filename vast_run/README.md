# vast.ai run book — YAM pi0-FAST teleop-oversampling co-fine-tune

Two LeRobot v2.1 datasets, stored at **~33% teleop / 67% ego** by frames, sampled
at a **fixed per-batch ratio** so teleop's gradient share is set explicitly:

| source | repo | frames | per batch (E-A) |
| --- | --- | --- | --- |
| teleop (real robot) | `angkul07/abc-teleop` | ~540k | 32 / 64 |
| ego (retargeted) | `angkul07/EgoDex-PickPlace-YAM-14dof-multiview` | ~1080k | 32 / 64 |

Both datasets are already v2.1 — nothing is converted, `LeRobotDataset` reads
them from `$HF_LEROBOT_HOME/<repo_id>` directly.

## Configs

| config | p_teleop | batch split T/E | steps | warmup |
| --- | --- | --- | --- | --- |
| `pi0_fast_yam_mix_ea` | 0.50 | 32 / 32 | 50,000 | 1,000 |
| `pi0_fast_yam_mix_eb` | 0.625 | 40 / 24 | 50,000 | 1,000 |
| `pi0_fast_yam_mix_ec` | 0.50 | 32 / 32 | 100,000 | 2,000 |

Everything else matches the previous 7k run: LoRA `gemma_2b_lora`, batch 64,
peak LR 3.5e-5 → 3.5e-6 cosine (decay_steps == num_train_steps), bf16, EMA off,
QUANTILES norm, delta joints + absolute grippers.

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
- **Validation split is currently OFF** — every teleop episode is trained on. When
  you pick the held-out episodes on the box, list their indices in the teleop
  `MixtureSource`:
  ```python
  MixtureSource(repo_id="angkul07/abc-teleop", samples_per_batch=32,
                exclude_episodes=(3, 17, 42))
  ```
  They drop out of the training stream with nothing moved on disk, and the count is
  printed at startup. (`holdout_fraction=0.1, holdout_seed=0` is the alternative —
  a deterministic random split matching fidelity-sdk's `HoldoutSpec`.) Ego gets no
  holdout either way: headline metrics are teleop-only by design.
- **Augmentation is training-only** (`DataConfig.train_only_transforms`), so it is
  off at eval/serving by construction: photometric jitter on all 3 cameras,
  random crop-and-resize on the **top camera only** (wrist views are pose-coupled),
  no flips/rotations, no noise on state or actions.
- `run_yam.sh` refuses to start if `decay_steps != num_train_steps` or if the
  per-batch counts don't sum to the batch size.

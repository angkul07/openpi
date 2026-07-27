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
| `pi0_fast_yam7h_ea` | 0.50 | 32 / 32 | 23,600 | 500 |
| `pi0_fast_yam7h_eb` | 0.625 | 40 / 24 | 23,600 | 500 |
| `pi0_fast_yam7h_ec` | 0.50 | 32 / 32 | 47,200 | 950 |
| `pi05_yam7h_ea` | 0.50 | 32 / 32 | 23,600 | 500 |
| `pi05_yam7h_eb` | 0.625 | 40 / 24 | 23,600 | 500 |
| `pi05_yam7h_ec` | 0.50 | 32 / 32 | 47,200 | 950 |

Everything else matches the previous 7k run: LoRA `gemma_2b_lora`, batch 64,
peak LR 3.5e-5 → 3.5e-6 cosine (decay_steps == num_train_steps), bf16, EMA off,
QUANTILES norm, delta joints + absolute grippers.

## pi0.5 arms

`pi05_yam7h_*` runs the **same 7h mixture, same schedule, same data transforms**
against `Pi0Config(pi05=True)` — flow matching instead of FAST action tokens. The
data config is byte-identical to its `pi0_fast_yam7h_*` twin, so the only moving
part is the architecture. Four deltas, three of which are traps rather than tuning:

| | pi0-FAST | pi0.5 |
| --- | --- | --- |
| `action_dim` | 14 | **32** |
| `max_token_len` | 300 | **200** |
| `pi05` | — | **True** |
| `discrete_state_input` | — | **leave unset** (defaults to `pi05`, i.e. True) |
| weight loader | `pi0_fast_base` | **`pi05_base`** |

- **`action_dim=32` is mandatory.** `weight_loaders._merge_params` matches on key
  names and never compares shapes, so 14 does *not* raise at load — it dies later
  inside jit pointing somewhere unrelated. `YamOutputs` already slices `[..., :14]`
  and `PadStatesAndActions` zero-pads 14 → 32, so nothing downstream cares.
- **Do not copy `discrete_state_input=False` from `pi05_libero`.** With `pi05=True`
  `embed_suffix` emits no state token, so `False` means the model gets *no*
  proprioception at all — the discretized prompt is the only path in.
- **`max_token_len=200` is enough** because `TokenizePrompt` runs *before*
  `PadStatesAndActions` in `ModelTransformFactory`, so it tokenizes the real 14
  state values, not 32 padded ones (~90 tokens in practice).
- **The YAM-specific FAST tokenizer is dead here.** π0.5 uses `PaligemmaTokenizer`
  for text only; `Pi0Config` has no `fast_model_tokenizer` field.

**Norm stats carry over unchanged — copy, don't recompute.**
`compute_norm_stats.py` applies `repack + data_transforms` only (never
`model_transforms`), and `use_quantile_norm` is `model_type != PI0`, which is True
for `PI0_FAST` and `PI05` alike. Identical 14-dim quantile stats:

```bash
mkdir -p assets/pi05_yam7h_ea assets/pi05_yam7h_eb assets/pi05_yam7h_ec
cp -r assets/pi0_fast_yam7h_ea/yam7h_p50  assets/pi05_yam7h_ea/
cp -r assets/pi0_fast_yam7h_eb/yam7h_p625 assets/pi05_yam7h_eb/
cp -r assets/pi0_fast_yam7h_ec/yam7h_p50  assets/pi05_yam7h_ec/
```

Stage `[1/3]` then prints "already present" and skips. If it starts computing, you
skipped the copy — not fatal, just ~10 wasted minutes.

**Two pi0.5-only tools, both under `vast_run/pi05/`:**

```bash
# BLOCKING (exit 2 = do not launch): config shape, trainable split,
# prompt length on real task strings, norm-stat presence/shape
uv run vast_run/pi05/pi05_preflight.py pi05_yam7h_ea

# Read this BEFORE committing GPU-hours: per-source tracking residual,
# normalized action distribution, clip atoms, gripper mode collision
uv run vast_run/pi05/mixture_diagnostics.py \
  --norm-stats assets/pi05_yam7h_ea/yam7h_p50/norm_stats.json

# Ego gripper rescale -- REQUIRED before the first pi0.5 run on a fresh dataset.
# Dry run first; --revert undoes it; a sentinel blocks double-application.
uv run vast_run/pi05/rescale_ego_gripper.py            # dry run
uv run vast_run/pi05/rescale_ego_gripper.py --apply
rm -rf assets/pi05_yam7h_ea/yam7h_p50                  # stats are now stale
uv run scripts/compute_norm_stats.py --config-name pi05_yam7h_ea \
  --max-frames 200000 --skip-videos
```

### What the diagnostics actually said on the 7h mixture

Measured, not predicted — three of these contradict the original plan.

**Gripper collision was real and is now fixed.** Before the rescale, ego's "open"
sat *closer to teleop's closed* than to teleop's open:

| | teleop open | ego open | gap | after fix |
| --- | --- | --- | --- | --- |
| `R_grip` | +0.984 | −0.114 | **1.10** | ego +0.689 → gap **0.295** |
| `L_grip` | +0.986 | −0.216 | **1.20** | ego +0.819 → gap **0.167** |

Both are now well under the 0.5 mode-averaging threshold and the diagnostic's
warning no longer fires.

**Ego's gradient share is HIGHER than its sampling share, not lower.** The original
analysis predicted ego's narrow deltas would compress its targets toward zero and
weaken its contribution. The opposite is true: median arm-dim std is **teleop 0.215
vs ego 0.387**, so ego's normalized actions are ~1.8× *wider*, and ego clips past
|x|>1 on 2.6–7.8% of values against teleop's ~0–4%. Read E-A vs E-B accordingly —
`p_teleop=0.5` under-weights teleop in effective terms.

**Clip atoms are a non-issue.** All 24 detected atoms land at |normalized| ≤ 0.29,
so q01/q99 are *not* being set by the ±0.1/±0.2 retargeting clips. The script
prints its generic "consider teleop-only stats" note whenever any atom exists —
check the per-atom "inside range" verdicts before acting on it.

**The two sources teach different functions, sharply.** Ego's tracking residual is
exactly 0.000 on every dim (`action[t] ≡ state[t+1]`); teleop's median arm lead
ratio is **14.8**. Ego contributes no corrective signal at all.

**Prompt-level domain shortcut.** Teleop has **1** distinct task string; ego has
**471**. Under π0.5 the prompt *is* the conditioning, so source identity is
perfectly recoverable from the prompt alone — a more direct shortcut than the
wrist-blur one, and it is not addressed by image augmentation.

**Cost and memory.** MEASURED on 2× A100-80GB at batch 64: **3.53 s/step**, i.e.
**~23.1 h for 23,600 steps** (E-A / E-B) and **~46 h for 47,200** (E-C). Budget the
instance rental against 23 h — the pre-run estimate of 15–19 h was optimistic. GPU
utilization is 95%, so this is compute-bound and loader tuning will not move it. The
action expert trains full-rank, so the trainable set goes ~448M → **872.8M** and
checkpoints grow ~25–30% — drop `max_to_keep` to 2 if the checkpoint volume is
tight. Measured breakdown from `pi05_preflight.py` (`jax.eval_shape`, not an
estimate): action expert 427.9M + SigLIP 414.8M + LoRA 27.9M + projections 2.2M
trainable, Gemma 2B trunk 2508.5M frozen, ≈10.5 GB/GPU of AdamW moments and grads
under `--fsdp-devices 1`. Note the expert is 427.9M, not the 311M the
`gemma_300m` name suggests — that name counts the transformer stack only.

**The pi0.5 arms run `num_workers=16`. Do not bother raising it — measured, this
run is compute-bound, not loader-bound.** 200-step A/B on 2× A100-80GB, page cache
pre-warmed so run order could not decide it, steady-state windows only (steps
50–175, excluding XLA compile and loader spin-up):

| `num_workers` | s/step | GPU util | 23.6k ETA |
| --- | --- | --- | --- |
| 16 | **3.528** | 95.0% | 23.1 h |
| 32 | 3.545 | 94.7% | 23.2 h |

Doubling workers made it **0.5% slower**, and the within-config window spread is
0.7–0.9% — so the difference sits *below* the noise floor. At 95% GPU utilization
there is no loader headroom left to reclaim; three-camera decode is comfortably
keeping both GPUs fed. `param_norm` was bit-identical across the two runs at step
175, confirming the knob does not perturb the optimization.

Whether 8 would also suffice was not tested — 16 is known-sufficient and costs
nothing, so it stays. The lever for speed here is GPUs or batch size, not workers.

**Reading the loss curve.** `compute_loss` averages squared error over all 32
action dims, 18 of which are zero-padding where the target is recoverable as
`x_t / t`. The sharp drop in the first few hundred steps is mostly those dims, not
the task. Watch `grad_norm` over the first 500 steps against the ~2.5 seen on
pi0-FAST; if it runs hot, the fallback is `action_expert_variant="gemma_300m_lora"`
(rank 32) set on **both** the model and the `freeze_filter`.

W&B project defaults to `pi05-yam` for these arms (`pi0-fast-modal` otherwise);
override with `WANDB_PROJECT=...`.

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

For a `pi05_yam7h_*` arm, insert the norm-stat copy and the two pi0.5 checks from
the section above between steps 2 and 3; steps 3 and 4 are otherwise unchanged
(`run_yam.sh` and `upload_ckpt.py` both handle the prefix).

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

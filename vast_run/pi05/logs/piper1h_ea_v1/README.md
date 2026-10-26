---
license: apache-2.0
library_name: openpi
pipeline_tag: robotics
tags:
  - robotics
  - openpi
  - vla
  - pi0
  - pi05
  - flow-matching
  - behavior-cloning
  - piper
datasets:
  - angkul07/piper-h-ego-teleop-v21
---

# pi05-piper1h-ea — π₀.₅ finetuned on a 1-hour Piper H mixture

Training checkpoint for arm E-A of the Piper H 1-hour mixture study: π₀.₅ (flow matching) finetuned
from `pi05_base` on a 34/66 mixture (frame-wise) of real Piper H teleop and retargeted EgoDex human
video, for a 14-DoF Piper H robot.

This repo contains a raw Orbax training checkpoint (params + train_state + norm stats), not a
converted HF-format model. It is meant to be loaded by openpi.

**NOTE ON "BIMANUAL":** the rig is bimanual and the tensors are 14-D, but **action dims 7–13 are
identically zero in both halves of the data** — min and max are exactly 0.0. This is a single-arm
capture (the right arm, matching the `pick-right-*` session name) padded into a bimanual container.
Only 7 of the 32 model action dims carry any signal. Read every loss number below with that in mind.

---

## ⚠️ Read this first

| | |
|---|---|
| **Checkpoint here** | `2399` (final) only |
| **Evaluation** | **None.** No success rate, no rollout. Every number below is training loss. |
| **Teleop holdout** | 15 of 154 episodes withheld from training — but **never scored**. |

The run trained the full 2,400 steps and completed cleanly (`EXIT=0`, 2 h 02 m 59 s). Checkpoints
1000, 2000 and 2250 were written during the run but are **not uploaded here** — only the final one is.

Because there is no evaluation, **"final" means last, not best.** Nothing in this repo establishes
that 2399 outperforms 1000 on any task. Unlike most runs in this series, a holdout split *does*
exist (15 teleop episodes, `holdout_fraction=0.1`, seed 0), so an honest offline eval is possible —
it just has not been run.

**This is a deliberately light 2-epoch run.** It was configured to preserve base behaviour rather
than specialise hard. See *Learnings* §4 — the evidence says it succeeded at being light, which is
not the same as being good.

---

## Experiment

| Field | Value |
|---|---|
| Config | `pi05_piper1h_ea` (openpi TrainConfig) |
| Model | `Pi0Config(pi05=True)` — PaliGemma backbone + flow-matching action expert |
| PaliGemma variant | `gemma_2b_lora` |
| Init | `gs://openpi-assets/checkpoints/pi05_base/params` (12 GB) |
| Action dim | 32 (14 real, 18 zero-padded — **and 7 of the 14 are themselves constant zero**) |
| Action horizon | **30** (not the openpi default of 50 — see *Why H=30*) |
| Max token len | 200 |
| Hardware | 2× A100-SXM4-80GB (NVLink, NV12 / 12 bonded links), data-parallel, `--fsdp-devices 1` |
| Wall clock | **2 h 02 m 59 s** for 2,400 steps (≈3.07 s/step) |
| Code revision | `dd8fa13` (`angkul07/openpi`, branch `piper1h-pi05`) |
| W&B | project `piper-h-pi05`, run `piper1h_ea_v1` (`4ksr9cp0`) |

Throughput note: 3.07 s/step beat the 3.528 s/step recorded for 2× A100 SXM4 on the 7 h YAM arms.
That earlier box was power-capped at 275 of 400 W; this pair ran uncapped (measured 392–424 W draw,
61–64 °C, 100% util on both cards). Same silicon, ~14% more throughput, purely from the power cap.

## Hyperparameters

| Field | Value |
|---|---|
| Batch size | 64 (**24 teleop + 40 ego**, fixed per batch) |
| Steps | 2,400 |
| Optimizer | AdamW, `clip_gradient_norm=1.0` |
| LR schedule | Cosine, warmup 60 → peak 3.5e-5 → decay 3.5e-6 over 2,400 steps |
| EMA | off (`ema_decay=None`) |
| Dataloader workers | 16 |
| Checkpoint interval | every 250 steps, `max_to_keep=2`, `keep_period=1000` |
| Precision | fp32 params (`param_norm` tracked in fp32) |
| Env | `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` (required — batch 64 OOMs at the 0.75 default) |

The LR schedule was inherited unchanged from the YAM arms of this series so the mixture and the
embodiment are the only moving variables. It was not tuned for Piper.

`max_to_keep=2` / `keep_period=1000` were set from measured disk on the target box (150 GB overlay,
nothing else writable). π₀.₅ checkpoints turned out to be **13 GB** each, not the 16–17 GB budgeted;
the run peaked at 74 GB of 150 GB with 4 checkpoints live.

### Why H=30 and not the default 50

Two independent reasons, either sufficient:

1. **openpi's LeRobot path clamps `delta_timestamps` at episode ends** — it repeats the final action
   rather than dropping the sample — and **nothing in openpi consumes `is_pad`**. So clamped steps
   enter the flow-matching loss as genuine regression targets, not ignored padding. The clamped share
   of supervised action steps is `(H−1)/2L`; at ego's mean episode length of 78 frames that is
   **31.3% at H=50 versus 18.5% at H=30**, and at ego's median length of 60 it is 40.8% vs 24.2%.
   This is signal corruption, not reweighting.
2. **20 Hz.** H=30 at 20 Hz is 1.5 s, the same physical horizon that H=50 gave the 30 Hz YAM arms
   (1.67 s). Inheriting 50 here would have silently asked for a 2.5 s chunk.

## Data

A 1-hour mixture of two sources, sampled at a fixed **24/40 per batch** (`StratifiedBatchSampler` —
every batch has exactly 24 teleop and 40 ego, not probabilistic sampling).

| Source | Train frames | % frames | Per batch | Gradient share | Epochs over the run |
|---|---|---|---|---|---|
| Piper H teleop (real) | 23,347 | 32.7% | 24 | 37.5% | **2.47** |
| EgoDex → Piper H (retargeted) | 47,953 | 67.3% | 40 | 62.5% | **2.00** |
| **Total** | **71,300** | 100 | 64 | | 153,600 frames seen · 2.15 epochs |

Both halves live in `angkul07/piper-h-ego-teleop-v21`. The `repo_id` strings in the config
(`angkul07/piper-h-teleop-v21`, `angkul07/piper-h-ego-v21`) are **labels only** — the actual data is
resolved from the local `root` paths, so those two repos need not exist.

Teleop is 154 episodes / 25,075 frames before the holdout; **15 episodes** (`holdout_fraction=0.1`,
seed 0 — indices 2, 5, 10, 25, 38, 44, 72, 76, 89, 92, …) are withheld, leaving 23,347 train frames.
Ego is 613 episodes / 47,953 frames with no holdout. Both are LeRobot v2.1 at **20 fps**, 14-D state
and action.

Teleop's gradient share (37.5%) is **1.15× its storage share** — a deliberate but mild oversample
toward the real-robot half. The break-even (proportional) draw is 20.5/43.5; anything above 21 teleop
samples oversamples it. An even 32/32 draw was rejected because the pools are only 2.05× apart, and
at 32/32 ego reaches 2.0 epochs only at 3,000 steps, by which point teleop is at 4.24 — the fixed-count
sampler gives both sources the same frames per step, so one draw fixes one epoch ratio.

### Observation space

**Two real cameras, not three.** The rig stores three, but the key names do not describe their content —
this was established by extracting frames and looking, not by reading metadata:

| dataset key | teleop content | ego content | model slot |
|---|---|---|---|
| `front` | top-down workspace | full egocentric frame | `base_0_rgb` — **real** |
| `right` | right-arm wrist camera | tight crop on grasp point | `left_wrist_0_rgb` — **real** |
| `top` | **sideways view of the robot** | wide crop on grasp point | **dropped** |
| — | *(no camera)* | *(no camera)* | `right_wrist_0_rgb` = zeros, **masked off** |

`front` and `right` genuinely correspond across the two halves. `top` cannot: a head-mounted ego
camera can never produce a sideways view of the robot, so that key holds two unrelated images under
one name. Feeding it would make the halves disagree about what the slot means.

It is **dropped in the repack** rather than masked, because `Pi0.embed_prefix` runs SigLIP on every
entry of `obs.images` regardless of the mask — masking costs the same compute while keeping the
contradictory content. The surviving wrist view goes in **slot 1** with slot 2 as the masked padding
slot, matching `droid_policy` and `libero_policy`; `IMAGE_KEYS` is ordered and position carries
signal, so filling slots 0–1 and masking the trailing one is the on-distribution layout.

Caveat: dropping `top` at the repack does **not** skip its video decode — LeRobotDataset decodes
every video feature before transforms run. The stream is still read and thrown away.

Also: 14-D joint state fed as a discrete state token (`TokenizePrompt(discrete_state_input=True)`,
the π₀.₅ convention), and a language prompt from the LeRobot task string (`prompt_from_task=True`).

### Action space

- 14-D, horizon 30, zero-padded to 32.
- `DeltaActions(mask=(True×6, False, True×6, False))` — the 12 arm-joint dims are predicted relative
  to current state; the 2 gripper dims stay absolute.
- **Verified empirically that both halves store absolute actions**, which is what this mask assumes:
  `mean|action − state|` is 0.005 (teleop) and 0.022 (ego) against `mean|action|` of 0.73 and 1.16.
- The mask is symmetric per arm (6 joints + 1 gripper, twice), so it is correct regardless of which
  arm occupies dims 0–6 — which matters, because the active arm here is the **right** one, i.e. the
  layout is `[R j0-5, R grip, L j0-5, L grip]`.

**Normalization** — quantile (`use_quantile_norm=True`, q01/q99), asset `piper1h_p50`, computed
through the same 24/40 sampler and shipped in the checkpoint at
`2399/assets/piper1h_p50/norm_stats.json`. Zero-variance dims are safe: both normalize paths carry a
`+ 1e-6` guard, so the 25 constant dims produce no NaN.

**Train-only augmentation** — `ImageAugment(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.03,
crop_scale=(0.95, 1.0))`, geometric crop applied to `base_0_rgb` only.

---

## Data preparation — what we changed

Two defects in the **teleop** half blocked training outright. Both were fixed locally on the training
box, not upstream, so a fresh pull of the dataset will reproduce them. Ego was clean on both counts.

### 1. Parquet metadata: `datasets` 4.x `"List"` → 3.x `"Sequence"`

The teleop converter embedded HuggingFace feature metadata written by `datasets` ≥ 4.x, which renamed
`Sequence` to `List`. openpi pins `datasets==3.6.0`, whose `Features.from_dict` has no `List` type, so
`load_dataset()` raised before training began:

```
ValueError: Feature type 'List' not found. Available: ['Value', 'ClassLabel', ..., 'Sequence', ...]
```

Fixed by rewriting `"_type": "List"` → `"Sequence"` in the schema metadata of all **154** teleop
parquet files. Only the metadata string changed; every file was verified for identical row count and
bit-identical `observation.state` / `action` values before the swap. Ego carried no `huggingface`
metadata at all (only `pandas`), so its types are inferred from the arrow schema and it was unaffected.

### 2. Video PTS rebase: a uniform one-frame timestamp offset

Every one of the **462** teleop videos (154 episodes × 3 cameras) started at pts `0.05` instead of
`0.0` — exactly one frame at 20 Hz. All 613 ego videos started correctly at `0.0`. With lerobot's
default `tolerance_s=1e-4` this is a hard crash mid-training:

```
AssertionError: query timestamps violate tolerance (0.0500 > tolerance_s=0.0001)
queried: 0.0   loaded: 0.05
```

**No frames were missing.** Teleop frame counts matched parquet rows exactly (195/195, 188/188,
174/174), spacing was a uniform 0.05, and the last frame sat `+0.05` past the last parquet timestamp
(9.750 vs 9.700). The entire PTS timeline was shifted by one frame — the encoder started its clock at
`1/fps` rather than `0`.

Fixed losslessly with `ffmpeg -c copy -bsf:v setts=PTS-STARTPTS` (no re-encode). All 462 were then
verified: frame count preserved, first pts `0.0`, last pts equal to the episode's last parquet
timestamp. Originals retained alongside.

**Why rebase rather than widen `tolerance_s`.** Raising the tolerance to ~0.05 would have "worked"
with a one-line change, but it maps dataset frames 0 *and* 1 onto video frame 0, leaving every teleop
sample with a one-frame-stale image while ego stays correctly aligned. The two halves of the mixture
would then disagree about image/action alignment — the exact class of silent inconsistency that
quietly degrades a co-trained policy. Since frame counts were intact, the honest fix was available.

Note for anyone repeating this: `-itsoffset -0.05` looks equivalent and **silently drops a frame**
(194 vs 195, measured). Use `setts`.

Also worth setting explicitly: `tolerance_s` is `null` in both `meta/info.json`, so lerobot falls back
to 1e-4. That strictness is why this surfaced as a clean crash instead of silent misalignment — good,
but currently accidental rather than chosen.

### 3. No gripper rescale — measured as unnecessary

The YAM arms of this series needed a gripper range alignment, because ego's "open" normalized closer
to teleop's *closed*. **That is not the case here, and it was checked rather than assumed.** Both
halves store strictly binary `{0.0, 1.0}` grippers with the same polarity:

| | start | mid | end | fraction at 1 |
|---|---|---|---|---|
| teleop | 0.745 | 0.677 | **0.083** | 0.50 |
| ego | 0.472 | 0.466 | **0.411** | 0.44 |

Same encoding, no inversion, so no rescale was applied. But see *Caveats* — the ego gripper carries
almost no temporal signal, which is a different and arguably worse problem than a scale mismatch.

### 4. Norm stats

Computed after the two fixes above, over the post-fix mixture through the 24/40 sampler
(`--max-frames 200000 --skip-videos`), and verified free of NaN/inf on all 14 dims of both
`state` and `actions`. The video PTS fix does not affect them (`--skip-videos` reads only
state/action), so no recomputation was needed after the rebase.

### What was deliberately not changed

- **Ego retargeting error was not filtered.** Mean IK error is 4.770 cm over 613 clips, and it is
  **bimodal, not uniform**: 0.701 cm on clips where the wrist-orientation constraint is never pinned,
  versus 6.944 cm where it is pinned in >50% of frames. The cause is Piper's joint5 range (±1.22 rad)
  against the ±1.571 the source retarget assumed. `_PIPER1H_EGO_EXCLUDE` is empty — the per-episode
  pinning fraction that would drive a principled filter has not been exported.
- **No resolution matching between halves.** Teleop is 640×480 portrait, ego is 224×224 square; the
  resize to 224 squashes teleop ~1.33× vertically. Squash-vs-crop remains an open choice (`ROT90_K=0`).
- **Teleop was not rescaled or resampled.** It is the reference.

---

## Results

Training loss is the flow-matching objective, mean over dims of `(v_θ(x_t, t) − u_t)²`.

| Step window | Loss | chunk_first | chunk_last | grad_norm | param_norm |
|---|---|---|---|---|---|
| 0 – 100 | 0.04832 | 0.03718 | 0.07005 | 0.20740 | 1803.16 |
| 100 – 250 | 0.02526 | 0.00909 | 0.04255 | 0.09605 | 1803.18 |
| 250 – 500 | 0.02232 | 0.00819 | 0.03783 | 0.09034 | 1803.21 |
| 500 – 1,000 | 0.01893 | 0.00666 | 0.03138 | 0.08734 | 1803.28 |
| 1,000 – 1,500 | 0.01540 | 0.00577 | 0.02495 | 0.08917 | 1803.33 |
| 1,500 – 2,000 | 0.01286 | 0.00519 | 0.02074 | 0.08621 | 1803.36 |
| 2,000 – 2,400 | 0.01170 | 0.00489 | 0.01844 | 0.08489 | 1803.36 |

Per checkpoint written during the run (200-step trailing mean):

| Checkpoint | Loss | chunk_first | chunk_last | grad_norm | uploaded |
|---|---|---|---|---|---|
| 1000 | 0.01781 | 0.00635 | 0.02980 | 0.08502 | no |
| 2000 | 0.01235 | 0.00536 | 0.01983 | 0.08521 | no |
| 2250 | 0.01179 | 0.00530 | 0.01881 | 0.08345 | no |
| **2399 (final)** | **0.01142** | **0.00455** | **0.01788** | **0.08374** | **yes** |

Other figures: first-step loss 0.19416; minimum single-step loss 0.00741 at step 2,182; last-500-step
loss 0.01176 ± 0.00181; total `param_norm` drift over the whole run **+0.2074 (+0.0115%)**.

Gradient clipping (`clip_gradient_norm=1.0`) bound on exactly **3 steps — 0, 1 and 2** (grad_norm
1.170, 1.244, 1.052) and never again. Max grad_norm after step 2 is below 1.0 for the remaining 2,397
steps.

Full per-step metrics for all 2,400 steps are in `train_metrics_full.log`. The console log, including
data-loader and sampler diagnostics, is in `train_console.log`.

---

## Learnings and takeaways

**1. The absolute loss is even less interpretable here than usual, because 25 of 32 action dims are
constant zero.** 18 are the standard zero padding, where `x_t = t·noise` makes the target
`u_t = noise = x_t/t` a closed-form function of the input — free to fit. On top of that, **7 more are
the all-zero left arm**. So roughly 78% of the objective is trivially predictable. The headline loss
falling 0.194 → 0.048 in the first 100 steps is mostly the model learning to emit zeros. Compare
π₀.₅ runs to each other, never to an absolute threshold.

**2. `chunk_last` is the metric that carries information; `chunk_first` is nearly free.** At the end
of the run `chunk_last` (0.01788) is **3.93×** `chunk_first` (0.00455). Predicting the action at t=0
is close to trivial when actions are deltas from current state, so `chunk_first` collapsing to 0.005
says almost nothing. The 1.5 s-ahead prediction is where the task difficulty lives, and it fell
0.07005 → 0.01844 across the run — a 3.8× reduction, still descending at 2,400.

**3. Optimization was uneventful and the LR schedule is validated for this setup.** grad_norm settled
to ~0.085 by step 100 and stayed flat for the remaining 2,300 steps. Clipping bound only during the
first three steps — a warmup transient, not an ongoing constraint. No spikes, no instability.

**4. This run barely moved the base model, by design — and the evidence is unambiguous.** `param_norm`
drifted **+0.0115%** across the entire run (the 7 h YAM arm, ten times longer, drifted +0.12%). With
LoRA, a 3.5e-5 peak and ~2 epochs, essentially all capability here is inherited from `pi05_base`. That
was the stated intent of a conservative first arm, and it means the run is very unlikely to have
overfit. It equally means **you should not expect strong Piper-specific specialisation.** If eval comes
back bland rather than wrong, under-training is the first hypothesis, and more steps or a higher peak
LR is the lever — not more data.

**5. The run was not length-limited.** Loss fell monotonically in every window with no plateau, and
`chunk_last` was still descending at 2,400. 2,400 was chosen to hit ~2 epochs per source, not because
the curve flattened. Whether more steps buy real capability is unanswerable from this run.

**6. Aggregate loss cannot answer what this mixture was built to ask.** A single scalar over a 24/40
mixture cannot distinguish learning from the teleop half, the ego half, or fitting the 25 constant
dims. Per-source (`loss_teleop` / `loss_ego`) and real-vs-padded-dim splits remain the single
highest-value cheap change: split `jnp.square(v_t − u_t)` before the final mean, and carry a
`source_index` through `MixtureDataset` into the train step. Given that 78% of dims here are constant,
the real-vs-padded split matters more for this run than for any previous one.

**7. Check what the camera keys actually contain before mapping them.** The teleop rig's names are
actively misleading — `front` is the top-down view, `top` is a sideways view of the robot, and only
`right` is honest. Believing the names would have paired ego's head-mounted downward view against a
sideways robot view in the same slot. One frame extracted per key would have caught it; metadata
would not.

**8. Infrastructure notes worth reusing.**
- 2× A100 SXM4 **uncapped at 400 W ran 3.07 s/step vs 3.528 for an otherwise identical pair capped at
  275 W** — ~14% throughput for free. Check `nvidia-smi -q -d POWER` before benchmarking a rented box.
- `nproc` reported 128 on this container; the real quota from `/sys/fs/cgroup/cpu.max` was **30.7
  cores**. Size worker pools off cgroup, never `nproc`. `num_workers=16` remains a non-lever — the
  loader uses ~3 cores and the run is compute-bound.
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` is required for batch 64 on 80 GB; the 0.75 default OOMs.
- π₀.₅ checkpoints measured **13 GB** (6.7 GB params + 5.9 GB train_state), not the 16–17 GB budgeted
  from a +25–30% rule over π₀-FAST. Four live checkpoints peaked at 74 GB of a 150 GB overlay.
- A dataset authored with `datasets` ≥ 4.x will not load under openpi's pinned 3.6.0 if the converter
  embedded HF feature metadata. Cheapest check before renting a GPU: `pq.read_schema(f).metadata`.

---

## Caveats and open questions

These bound what the numbers above can support.

- **No evaluation of any kind.** A 15-episode teleop holdout exists and was correctly withheld
  (verified in the training log), but it was never scored. Loss curves only.
- **The data is single-arm in a bimanual container.** Dims 7–13 are exactly zero in both halves, so
  the model is learning 7 real dims out of 32. Any claim about "bimanual" capability is unsupported.
- **The ego gripper channel is close to noise, and ego is 62.5% of the gradient.** Teleop shows clean
  pick semantics (open 0.745 → closed 0.083); ego is flat at ~0.44 throughout with only a slight
  downward drift. Under π₀.₅'s few-step flow matching, a near-random gripper target on the majority of
  samples is precisely the setup that mode-averages into a half-open hand. **This is the first thing to
  suspect if grasping underperforms**, and the fix is upstream in the ego gripper labels, not in
  hyperparameters.
- **Most of the gradient comes from synthesized, retargeted data.** Ego's two used views are
  keypoint-guided crops of a single egocentric camera, and its actions are IK retargeting of human hand
  poses with a 4.770 cm mean residual. The sim-to-real gap of that 62.5% is unmeasured.
- **Language conditioning comes entirely from the ego half.** Teleop has 1 task string; ego has 100.
  `prompt_from_task=True` therefore conditions the ego half and does nothing for the half an eval
  would score.
- **Both data fixes were applied to the local copy, not upstream.** Pulling the dataset fresh gives
  back the `List` metadata and the 0.05 s PTS offset, and training will crash the same way. Fix the
  converter, or re-apply both steps.
- **The ego retarget error is high and unfiltered** (4.770 cm mean, bimodal 0.701 / 6.944 cm). No
  episodes were excluded.
- **Single run, single seed, no ablation.** Nothing here isolates the mixture ratio, the horizon
  choice, or the two-camera decision.

---

## Repo layout

```
2399/
  _CHECKPOINT_METADATA
  params/                          # Orbax OCDBT — inference weights (6.7 GB)
  train_state/                     # optimizer state — for resuming only (5.9 GB)
  assets/piper1h_p50/norm_stats.json
train_metrics_full.log             # complete per-step metrics, steps 0–2399
train_console.log                  # console log incl. data-loader / sampler diagnostics
```

The checkpoint is ~13 GB total. For inference you only need `params/` and `assets/` — `train_state/`
is ~46% of the bytes and is needed only to resume training.

## Usage

```bash
# inference only — skipping train_state roughly halves the download
hf download angkul07/pi05-piper1h-ea \
  --include "2399/params/*" "2399/assets/*" "2399/_CHECKPOINT_METADATA" \
  --local-dir ./pi05-piper1h-ea
```

```python
from openpi.policies import policy_config
from openpi.training import config as _config

cfg = _config.get_config("pi05_piper1h_ea")
policy = policy_config.create_trained_policy(cfg, "./pi05-piper1h-ea/2399")

action_chunk = policy.infer({
    "observation/front_image": front_rgb,   # HxWx3 uint8 — top-down / egocentric view
    "observation/right_image": right_rgb,   # HxWx3 uint8 — wrist / grasp-point view
    "observation/state":       state_14d,
    "prompt":                  "pick up the block and place it in the bin",
})["actions"]                               # (30, 14)
```

Note the observation keys: **two images, not three.** `observation/top_image` is not consumed —
`PiperInputs` reads only `front` and `right`, and fills `right_wrist_0_rgb` with a masked zero array.
Expect dims 7–13 of the returned actions to be ~zero, since they were constant in training.

Requires the `pi05_piper1h_ea` config, which lives on branch `piper1h-pi05` of `angkul07/openpi`
(revision `dd8fa13`), along with `src/openpi/policies/piper_policy.py`.

To resume training, point openpi at the parent directory:

```bash
uv run scripts/train.py pi05_piper1h_ea --exp-name piper1h_ea_v1 --fsdp-devices 1 --resume
```

## License

Apache-2.0, inherited from openpi and `pi05_base`. The underlying datasets carry their own terms; the
EgoDex-derived half inherits EgoDex's license.

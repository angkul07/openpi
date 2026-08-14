# CHANGES — what this fork adds on top of upstream openpi

Fork base: **`15a9616`** (`update output objects to support batching`, upstream
`Physical-Intelligence/openpi`). Everything below is ours.

This file exists because the work is spread across ~900 lines of added config in a
single upstream file plus a `vast_run/` toolchain, and neither is discoverable from
`git log` alone. Read this before adding another training arm.

**Branch state (as of this merge):** `main` == `abcego-screwdriver-pi05-teleop`,
both at `b157cd5`. `piper1h-pi05` is still separate and carries three more configs
(see [Branch map](#branch-map)).

---

## 1. Why the fork exists at all

Upstream openpi trains one dataset per config. Every experiment we run is a
**mixture** — real teleop plus retargeted ego data — where the *gradient share per
source* is the independent variable. That single requirement is what forced all the
core changes below:

- storage ratio ≠ sampling ratio, so a **fixed per-batch draw** was needed;
- norm stats had to be computed **through that same sampler**, not over raw storage;
- an **offline holdout** had to be withheld by index without moving files;
- runs are long and rented, so **resume had to seek the data stream**, not replay it.

Everything else (augmentation opt-out, per-step metrics, rolling checkpoints) is
support for reading the results honestly.

---

## 2. Changes to upstream files

| File | Change | Why |
| --- | --- | --- |
| `src/openpi/training/data_loader.py` (+379) | `MixtureDataset`, `StratifiedBatchSampler`, `make_stratified_batch_sampler`, `_NoVideoLeRobotDataset`, `select_holdout_episodes`, `_remap_episode_data_index` | The core of the fork. Concatenates N LeRobot datasets and draws a **fixed count per source per batch**, so `p_teleop` is set explicitly and never inherited from disk. Also seeks by `start_batch` for resume. |
| `src/openpi/training/config.py` (+883) | `MixtureSource`, `LeRobotRobotDataConfig` / `LeRobotRobotMixtureDataConfig`, `TrainConfig.max_to_keep`, and 14 training arms | The config surface for the above, plus every experiment. **This is the file that has gotten unmanageable** — see [§6](#6-the-config-problem). |
| `src/openpi/training/augment.py` (+132, new) | `ImageAugmentConfig` / `ImageAugment`, wired through `DataConfig.train_only_transforms` | Data-side augmentation that is off at eval by construction. |
| `src/openpi/models/pi0_config.py` (+10) | `image_augmentation: bool = True` | Opt-out for openpi's **built-in, separate** model-side augmentation. Defaults `True` so no existing config changes behaviour. |
| `src/openpi/models/pi0.py` (+9) | `preprocess_observation(train=train and self.image_augmentation)` | Implements that opt-out. `train` gates nothing else in `compute_loss`, so this disables augmentation and nothing more. |
| `scripts/train.py` (+78) | Flow-loss reporting (`flow_loss`, `flow_loss_chunk_first`, `flow_loss_chunk_last`); per-step `train_metrics.log`; resume seeks the data stream to `latest_step + 1` | `loss` means cross-entropy on FAST arms and flow MSE on pi0.5 arms — logging both under one name made the two families unreadable side by side. `chunk_last` is the metric that actually moves. |
| `scripts/compute_norm_stats.py` (+52) | `--skip-videos`; mixture stats computed **through the training sampler**; output keyed on `asset_id` not `repo_id` | Raw-storage stats would be ego-dominated (33/67) and would misnormalise exactly the teleop grippers the experiments prioritise. `--skip-videos` turns hours into minutes — the script only reads `state`/`actions`. |
| `src/openpi/training/checkpoints.py` (+11) | `max_to_keep` plumbed through (was hardcoded `1`) | A rolling window of recent checkpoints, so a run can be scored at several points without keeping 50 × 10 GB. |
| `src/openpi/policies/robot_policy.py` (new) | `RobotSpec` + `RobotInputs` / `RobotOutputs`, parameterised by embodiment | Replaced the per-robot `yam_policy.py` / `piper_policy.py` pair, which were ~95% identical. Slices to the spec's action width, which is what makes `action_dim=32` safe on pi0.5. |
| `configs/fd/teleop_holdout.json` (+261, new) | 249 held-out `abc-teleop` episode indices | The offline eval set. Nothing moves on disk — these are simply never sampled. |

### Invariants these introduced

- `decay_steps` **must** equal `num_train_steps`. openpi defaults `decay_steps` to
  30k independently, so a short run silently never decays. `run_yam.sh` stage [0]
  asserts this.
- Norm stats are **per-ratio and per-distribution**, keyed by `asset_id`. Two arms
  with the same sources but a different draw need different stats.
  `run_yam.sh` skips computation when the file exists, so a stray copied directory
  normalises against the wrong distribution *silently*.
- `exclude_episodes` holds **original** dataset indices. Pre-selected subsets
  (7h, 10/90) are renumbered `0..N` and already exclude the holdout physically —
  reusing the index list there is wrong twice over.

---

## 3. `vast_run/` — the run toolchain (all new)

Everything needed to take a rented box from bare to training. Nothing here is
imported by openpi itself.

| Script | Role |
| --- | --- |
| `run_yam.sh` | The launcher. Stage [0] validates config shape (mixture non-empty, `sum(samples_per_batch) == batch_size`, `decay_steps == num_train_steps`, datasets are v2.1); [1/3] norm stats; [2/3] train; [3/3] upload. |
| `README.md` | The run book — per-arm tables, order of operations, and the "things that will bite you" list. |
| `dl.py` | Dataset fetch, resumable, waits out HF 429s. |
| `preflight_checks.py` | **Blocking** gripper-convention and delta-distribution check, parquet only. |
| `pi05/pi05_preflight.py` | pi0.5-specific: config shape, trainable split via `jax.eval_shape`, real prompt length, norm-stat presence. Exit 2 = do not launch. |
| `pi05/mixture_diagnostics.py` | Per-source tracking residual, normalised action distribution, clip atoms, gripper-mode collision. Read before spending GPU-hours. `--episodes` is **not** optional for reproducibility — the sampler is deterministic but strided. |
| `pi05/rescale_ego_gripper.py` | Reversible piecewise-linear ego→teleop gripper remap, with `.npz` sidecar and a double-application sentinel. Required before a first pi0.5 run on a fresh dataset. |
| `make_holdout.py` / `select_mixture.py` / `build_mixture.py` / `audit_mixture.py` | Carve the holdout, select the pre-sized subsets, build and verify the mixture manifest. |
| `mcap_to_lerobot.py` | MCAP → LeRobot v2.1 converter for `abc-ego`. Unifies two rigs (3-cam h264 848×480 vs 4-cam h265 1920×1200) onto one 3-camera schema and a uniform fps grid. Two-phase: parallel per-episode convert into `.staging/`, then a cheap serial assemble — so phase 1 is resumable while downloads finish. |
| `validate_lerobot_v21.py` | Structural validator for converter output. Catches the failures that break training *silently*: row-count vs `episodes.jsonl`, `timestamp == frame_index/fps` (LeRobot enforces to 1e-4 s), globally contiguous `index`, decodable videos with the right frame count, non-flat first frame, stats round-trip. |
| `upload_ckpt.py`, `resume_train.sh` | Push checkpoints; resume an interrupted run. |
| `yam7h_manifest.json`, `norm_stats.json` | The 7h selection manifest and a reference stat file. |

---

## 4. Training configs

All live in `configs/fd/yam/` (see [§6](#6-the-config-problem)). Batch 64,
`gemma_2b_lora`, cosine 3.5e-5 → 3.5e-6, EMA off throughout unless noted.

### pi0-FAST arms

| Config | Sources | Draw (T/E) | Steps | Purpose |
| --- | --- | --- | --- | --- |
| `pi0_fast_yam` | `Kavin60606/yam_pi0fast_train` | — | 30,000 | Full fine-tune. Needs 80 GB. Not the recommended first run. |
| `pi0_fast_yam_low_mem_finetune` | same | — | 7,000 | LoRA baseline. Undertrained by design (budget-fit). |
| `pi0_fast_yam_mix_ea` | abc-teleop + ego, 15h | 32/32 | 50,000 | Baseline ratio. |
| `pi0_fast_yam_mix_eb` | same | 40/24 | 50,000 | Teleop-biased (62.5% gradient share). |
| `pi0_fast_yam_mix_ec` | same | 32/32 | 100,000 | Long arm at the safe ratio. |
| `pi0_fast_yam7h_ea/eb/ec` | pre-selected 7.014h local subsets | 32/32, 40/24, 32/32 | 23,600 / 23,600 / 47,200 | Same experiment at 7h. E-B holds E-A's step count on purpose — the arms must differ in **mixing ratio alone**, so unequal optimisation budgets would confound it. |

### pi0.5 (flow-matching) arms

`pi05_yam7h_*` is byte-identical to its `pi0_fast_yam7h_*` twin on the data side, so
**architecture is the only moving part**. Four deltas, three of which are traps:

| | pi0-FAST | pi0.5 |
| --- | --- | --- |
| `action_dim` | 14 | **32** — `_merge_params` matches on key names and never compares shapes, so 14 does *not* raise at load; it dies later inside jit pointing nowhere near the cause. |
| `max_token_len` | 300 | **200** — 300 held FAST action tokens; pi0.5 has none. ~90 tokens in practice. |
| `discrete_state_input` | — | **leave unset**. Do not copy `False` from `pi05_libero`: with `pi05=True` the state token is already absent, so `False` means *no proprioception at all*. |
| weight loader | `pi0_fast_base` | `pi05_base` |
| `num_workers` | 8 | 16 (measured; 32 was 0.5% *slower* — below the noise floor) |

| Config | Sources | Draw | Steps | Purpose |
| --- | --- | --- | --- | --- |
| `pi05_yam7h_ea/eb/ec` | 7h teleop + ego, two roots | 32/32, 40/24, 32/32 | 23,600 / 23,600 / 47,200 | The pi0-FAST arms under flow matching. Norm stats are **reused** from the FAST twin (quantile stats are identical across `PI0_FAST` and `PI05`). |
| `pi05_yam1090_ea` | 10/90 teleop:ego by storage | 24/40 | 32,000 | Tests whether a small, high-quality teleop pool can be oversampled. 10.14 teleop epochs, so overfitting is the failure mode — hence `keep_period=5_000` to locate the knee after the fact. **Fresh stats** (`yam1090_p375`). |
| `pi05_abcego_sd` | `abc-ego` screwdriver, 730,496 frames | 64/0 | 11,414 | 100% teleop, single task, **exactly one epoch**, no augmentation. |
| `pi05_50run_ea` | `angkul07/50_run_v21_fixed`, 50/50 pre-merged | 64 | 17,700 | The openpi counterpart of the LeRobot 50/50 run. Same data, schedule and trainable set, so the two are directly comparable. |

#### "Mixture of one" — why single-source configs still use `LeRobotRobotMixtureDataConfig`

`pi05_abcego_sd` and `pi05_50run_ea` have one source each and still go through the
mixture path, deliberately, for two mechanical reasons:

1. `create_torch_dataset()` hardcodes `root=None` off the mixture path, so the dataset
   would have to live at `$HF_LEROBOT_HOME/<repo_id>` and be symlinked into place.
   `MixtureSource` carries `root`, so data stays where the converter wrote it.
2. `run_yam.sh` stage [0] asserts `data.mixture` is non-empty — a single-source config
   fails the launcher outright.

Sampling is unaffected: one source draws all 64 from a reshuffled permutation, i.e.
ordinary shuffled training.

#### Augmentation is two independent stacks

There are **two**, and disabling one leaves the other running:

- data-side: `DataConfig.augment_config` → `ImageAugmentConfig` → `train_only_transforms`
- model-side: openpi's built-in stack inside `compute_loss`, gated by
  `Pi0Config.image_augmentation`

`pi05_abcego_sd` and `pi05_50run_ea` set **both** off. Also worth knowing when reading
a loss curve: augmentation touches **images only**. State and action streams repeat
verbatim on every revisit, so it blunts visual memorisation, not action memorisation.

---

## 5. Branch map

| Branch | Adds | Status |
| --- | --- | --- |
| `main` | Everything in §2–§3 plus the YAM pi0-FAST and `pi05_yam7h_*` arms | Merged target |
| `abcego-screwdriver-pi05-teleop` | `mcap_to_lerobot.py`, `validate_lerobot_v21.py`, `image_augmentation` opt-out, flow-loss reporting, and `pi05_abcego_sd` / `pi05_50run_ea` / `pi05_yam1090_ea` | **Merged into `main`** (fast-forward, `b157cd5`) |
| `piper1h-pi05` | `pi05_piper1h_ea` (24/40, 2,400 steps), `pi05_piper1h_teleop` (64/0, 600 steps, control arm), `pi05_piper20m_ea` (32/32, 800 steps) — the Piper H 1-hour mixture | **Not merged.** Also carries uncommitted work adding `--merged-root` support to `rescale_ego_gripper.py`. |

---

## 6. The config problem, and the system that fixes it

Fourteen experiment arms (17 counting the unmerged Piper branch) had accumulated
inside `src/openpi/training/config.py`, an upstream file, as ~900 lines of
`TrainConfig` literals and comment blocks. Why it hurt:

- **Merge surface.** Every branch edited the same list in the same file, so every
  rebase onto upstream touched it.
- **No isolation.** A Piper experiment and a YAM experiment were neighbours in one
  list. Nothing scoped them apart.
- **The knowledge was in comments.** The reasoning that makes these arms correct —
  epoch math, why `action_dim=32`, which stats may be copied — sat as comment text
  next to the literal: untestable, unimportable, invisible to `git log`.
- **Cross-arm coupling by convention only.** "`decay_steps` must equal
  `num_train_steps`" was enforced by a bash assert in `run_yam.sh`, not by the config
  layer.

### What was built

A per-client config tree that **registers into** openpi instead of editing it. Full
documentation is in [`configs/README.md`](configs/README.md).

| Piece | What it does |
| --- | --- |
| `src/openpi/training/registry.py` | Walks `configs/`, imports every module, each registering its arms. Discovery is lazy (client modules import `config.py`, so import-time discovery would be a cycle). A module that fails to import is a **hard error** — a config that quietly vanishes from the CLI is how you launch the wrong arm. |
| `src/openpi/training/fingerprint.py` | Writes `norm_stats_fingerprint.json` beside `norm_stats.json` recording the sources, roots, draws and holdout the stats were computed through; `_load_norm_stats` checks it and **raises** on mismatch. Legacy stats with no fingerprint warn instead, so existing boxes keep working. |
| `configs/_shared/schedule.py` | `Schedule` derives `decay_steps` from `num_train_steps` — they cannot disagree — and computes step counts from epoch targets instead of hand-written tables. |
| `configs/_shared/arms.py` | `pi05_arm()` / `pi0_fast_arm()`. One `Pi0Config` feeds both `model` and `freeze_filter` (they used to be two hand-copied instances); one `augment=` flag moves both augmentation stacks; refuses draws that don't sum to `batch_size` and mixtures with no `asset_id`. |
| `configs/fd/` | Our own R&D arms — `datasets.py` (roots, frame counts, holdout), `yam/*.py` (one module per experiment, reasoning in the docstring). |
| `configs/_template/` | Skeleton to copy for a new client. |
| `scripts/dump_configs.py` | Canonical, address-scrubbed dump of every config including what `data.create()` produces. Dump before, dump after, diff. |

`config.py` shrank from 1,872 to 1,127 lines and now contains only upstream openpi's
own recipes plus the mixture machinery.

### What did not change

Config names, checkpoint paths, W&B history and `assets/<config>/` directories are all
untouched, so existing runs and stats stay valid. `scripts/dump_configs.py` proved the
migration byte-identical across **all 45 configs** — including each arm's fully
created `DataConfig` (transforms, mixture, delta masks). The `fd/` arms keep their
historical unprefixed names for that reason; new client arms must be prefixed with the
client slug, and registration raises on any collision.

### Still open

- Piper's 3 arms migrate when `piper1h-pi05` lands; that branch will conflict on
  `config.py` once, and the resolution is to delete its arms and re-add them under
  `configs/fd/piper/`.
- `run_yam.sh` derives its checkpoint subdirectory by stripping known family prefixes,
  so a new client's arms need the experiment name passed explicitly as `$2`.
- `LeRobotYamDataConfig` was robot-parameterised — see the row above. What remains
  embodiment-specific and is *not* yet in `RobotSpec`: the assumption of an arm-major
  `[joints..., gripper]` state layout (a dataset with different **per-arm** structure
  still needs its own mask; a right-arm-first layout is fine, since the mask is
  symmetric across arms), and control frequency, which nothing in the config models —
  the 20 Hz Piper data and the 30 Hz YAM data are indistinguishable to openpi.

  Image resolution is **not** on this list. `IMAGE_RESOLUTION = (224, 224)` is
  openpi's own model-level constant (`models/model.py`), and `preprocess_observation`
  resizes anything that does not match it for every config, upstream's included. It is
  fixed by the pretrained SigLIP input size, so it is a property of the model, not the
  robot, and does not belong in a robot spec. The only nit is cosmetic:
  `ModelTransformFactory` writes the literal `224, 224` three times instead of
  referencing the constant.

"""mf / Piper DUAL ARM: 100% teleop, ISR-standardized handover, three cameras.

`mf_pi05_teleop_isr` -- the teleop-only arm on the ISR-resampled bimanual handover
capture (`Kavin60606/bimanual-handover-isr-std`, see `configs/mf/datasets_isr.py`).
Same embodiment, task and recipe as `mf_pi05_dual_ea`, with two deliberate
differences: NO ego half (the whole batch is teleop), and the data has been through
ISR pacing normalization. What this run answers is what the recipe does on
standardized real-rig data alone -- it is the natural teleop baseline for any later
ISR-teleop + ego mixture.

BEFORE LAUNCHING -- see the datasets module for the full versions:
    1. The HF repo is LeRobot v3.0 with AV1 video; convert to v2.1 + h264 under
       `datasets_isr.ROOT` first. The loader cannot read it as published.
    2. Confirm the renumbered episodes 0..49 preserve recording order, or the tail
       holdout is no better than a scattered draw.
    3. Extract a frame from `observation.images.top` and check its orientation --
       this rig family has recorded it sideways before.

WHY 5,600 STEPS AT BATCH 128 (derived, not chosen)
    The run goes on 4x H100 SXM (data-parallel; `fsdp_devices` stays 1). Batch 128
    is 32 samples per GPU -- the same per-GPU footprint as every measured 2x
    A100-80GB run at batch 64, so memory is a solved question, as on the dual run.

    Exposure is TEN epochs, a deliberate step DOWN from the 14-15 passes every
    prior run of the recipe got: the pool is ~70.5k training frames of ONE task
    and 45 episodes -- a third of the dual mixture's per-half budget -- and 15
    passes over that little data is more revisits per episode than any prior run
    took. `Schedule.for_epochs` derives the steps: 10 x 70,541 / 128 -> 5,600
    after rounding (10.2 realised epochs; `describe()` prints the exact table).
    If the epoch call was wrong in either direction, the holdout ladder on
    episodes 45..49 is what says so.

    WALL-CLOCK, ESTIMATED AND NOT MEASURED: the dual run estimated 1.5-2.5 s/step
    for this exact shape (batch 128, three cameras, 4x H100), so 5,600 steps is
    ~2.5-4 h. Measure the first 200 steps on the box before trusting it.

PEAK LR, WARMUP, AND EVERYTHING ELSE INHERITED
    peak 3.5e-5 -> 3.5e-6 cosine (NOT rescaled for batch 128, for the dual run's
    reason: at the bigger batch it is effectively a halved per-sample LR,
    conservative in the safe direction), LoRA `gemma_2b_lora` trunk with full-rank
    action expert, EMA off, `action_dim=32` (mandatory padding -- see `pi05_arm`),
    `action_horizon=50`, `max_token_len=200`, augmentation on. Warmup is the same
    ABSOLUTE 1,000 steps -- now 17.9% of this very short run, easily the highest
    fraction of any arm, but the bound that matters is absolute: fresh LoRA
    B-matrices produce their largest gradients in the first few hundred steps
    whatever the run length. If anything about this run gets revisited first, this
    is the knob: a lazier-than-expected loss curve here is warmup eating a fifth
    of the schedule, before it is anything else.

    One ISR-specific caveat on `action_horizon=50`: the restamped 20 Hz makes the
    chunk 2.5 s NOMINAL, but pauses are collapsed, so a chunk spans more real
    motion than on raw Piper teleop. Not comparable to the dual run's chunks even
    though every number matches.

    Early stop is the same stall guard as the previous runs, same expectations: it
    likely never fires under cosine decay, and if it fires it leaves an un-annealed
    checkpoint. It decides when to stop paying; the holdout decides what to ship.

NUM_WORKERS=32 IS THE DUAL RUN'S GUESS FOR THIS EXACT SHAPE
    The measured 16 was batch 64, three h264 cameras, 2x A100, compute-bound. This
    run has the dual run's shape instead -- double the samples per step on GPUs
    roughly twice as fast -- so it inherits the dual run's sizing: the measured
    per-GPU worker count times the new GPU count. Same caveats too: A/B against 16
    in the first 200 steps, size against the cgroup quota rather than nproc, and
    ALL of it is void if AV1 survived into the v2.1 build instead of h264.

RUNNING IT
    `run_yam.sh` derives its checkpoint subdirectory by stripping a known family
    prefix, which `mf_` does not match, so pass the experiment name explicitly:

        ./vast_run/run_yam.sh mf_pi05_teleop_isr teleop_isr
"""

from __future__ import annotations

from configs._shared.arms import pi05_arm
from configs._shared.robots import PIPER_DUAL
from configs._shared.schedule import Schedule
from configs.mf import datasets_isr as ds
from openpi.training import registry
import openpi.training.config as _config

BATCH_SIZE = 128  # 4x H100, 32/GPU -- the measured per-GPU footprint. See docstring.

# Exposure is the thing held fixed (10 passes -- deliberately below the recipe's
# usual 14-15, sized to this small single-task pool); the step count falls out.
# See the docstring before changing either.
SCHEDULE = Schedule.for_epochs(
    frames=ds.TRAIN_FRAMES,
    batch_size=BATCH_SIZE,
    epochs=10,
    round_to=100,  # -> 5,600 steps; 500 would round a 5.5k run up by a whole 9%
    # Absolute, not the 2% default (which would give ~112 -- far shorter than the
    # measured peak-grad window). See the warmup note in the docstring.
    warmup_steps=1_000,
)

# Same stall/divergence guard as `mf_pi05_ea` and `mf_pi05_dual_ea`, unchanged. At
# this run length it is nearly decorative -- min_steps 5,000 of 5,600 leaves it a
# 600-step window to fire in -- but it still catches outright divergence, and
# changing its parameters per-run would make "same guard" mean nothing.
EARLY_STOP = _config.EarlyStop(
    metric="loss",
    patience_steps=3_000,
    min_rel_delta=1e-3,
    min_steps=5_000,
)

# Single source, whole batch -- but still through the mixture path, which is the only
# path (`create_torch_dataset()` hardcodes root=None off it; the builder asserts
# non-empty). The holdout exclusion is what makes this differ from "just train on the
# repo": episodes 45..49 never enter a batch.
SOURCES = (
    _config.MixtureSource(
        repo_id=ds.REPO,
        samples_per_batch=BATCH_SIZE,
        root=ds.ROOT,
        # The contiguous 5-episode tail (45..49), 10% of the pool -- the ladder's
        # scoring set, and this run's ONLY validation signal. See the datasets module.
        exclude_episodes=ds.HOLDOUT_EPISODES,
    ),
)


registry.register(
    pi05_arm(
        "mf_pi05_teleop_isr",
        robot=PIPER_DUAL,
        sources=SOURCES,
        # Its own asset_id, never shared: ISR changes the per-step delta distribution,
        # so these norm stats describe THIS pool and cannot be copied from
        # `mf_dual_ea` or from any raw-teleop run, even on the same rig.
        asset_id="mf_teleop_isr",
        schedule=SCHEDULE,
        batch_size=BATCH_SIZE,
        early_stop=EARLY_STOP,
        repo_id=ds.REPO,
        # The dual run's sizing for this exact shape (batch 128, 3 cameras, 4x
        # H100), a guess to be A/B'd against 16. See NUM_WORKERS above.
        num_workers=32,
        # Crash-resume granularity: ~15-40 min of work at risk at H100 step times.
        save_interval=1_000,
        # THE LADDER IS THE POINT: no in-training validation, so shipping = scoring
        # saved checkpoints against the 5-episode holdout afterwards. keep_period
        # must stay a multiple of save_interval, and on a 5,600-step run every
        # coarser period collapses the ladder (3,000 would pin ONE checkpoint), so
        # every save is pinned: 1k..5k = 5 pins (~65 GB at ~13 GB each) plus the
        # rolling last at 5,600 -- the same ladder density the longer runs got from
        # their coarser periods, and each pin is ~2 epochs apart, dense enough to
        # find the knee if even 10 epochs overfits this pool.
        max_to_keep=1,
        keep_period=1_000,
    )
)


def describe() -> str:
    """The epoch table. Reporting only -- nothing here feeds config.

    uv run python -c "from configs.mf.piper import teleop_isr as t; print(t.describe())"
    """
    per_source = (("teleop (ISR-std)", ds.TRAIN_FRAMES, BATCH_SIZE),)
    return "\n".join(
        [
            "mf_pi05_teleop_isr",
            SCHEDULE.describe(per_source),
            "",
            f"teleop holdout: episodes {ds.HOLDOUT_EPISODES[0]}..{ds.HOLDOUT_EPISODES[-1]}",
            f"early stop: {EARLY_STOP}",
            f"prompt/task: {ds.TASK}",
        ]
    )

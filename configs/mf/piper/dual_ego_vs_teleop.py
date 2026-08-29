"""mf / Piper DUAL ARM: 50/50 retargeted human video and real teleop, three cameras.

`mf_pi05_dual_ea` -- the bimanual successor to `mf_pi05_ea`. Same client, same
question (does retargeted human video help the real-rig task), new embodiment: both
Piper arms live (14-D, joints-major layout -- see `PIPER_DUAL`), three real camera
views, 1 h 55 m of teleop against 1 h 58 m of retargeted video under
`/workspace/final_data/`. Half of every batch from each half, as before, and this
time the halves are budget-matched within 2.3% on disk too.

BEFORE LAUNCHING -- the data is aligned (see `configs/mf/datasets_dual.py`), three
things remain, all cheap:
    1. `datasets_dual.TASK` is still "TBD": read the single teleop task string from
       `meta/tasks.jsonl` -- it is the prompt every eval must use.
    2. Confirm teleop episodes are stored in recording order, or the tail holdout is
       no better than a scattered draw.
    3. The 14-dim p1/p50/p99 preflight, teleop against ego. The schemas agree; the
       occupancy may not (wrist saturation, and the known gripper resting-aperture
       divergence -- ego near-closed, teleop open).

WHY BATCH 128 AND 30,000 STEPS, TOGETHER
    This run goes on 4x H100 SXM (data-parallel; `fsdp_devices` stays 1). Batch 128 is
    32 samples per GPU -- the same per-GPU footprint as every measured 2x A100-80GB run
    at batch 64, so memory is a solved question, not a hope.

    The step count is the other half of the same decision. The client's opening
    position was 60,000 steps, priced at batch 64: 3.84M samples. Keeping 60,000 AND
    doubling the batch would silently double that to ~29 passes over the corpus --
    twice the memorisation exposure and twice the rental, bought by accident. 30,000
    at 128 is the SAME 3.84M samples (~14-15 passes over each half; `describe()`
    prints the exact table), in about half the wall-clock. What is genuinely halved is
    the number of optimizer steps; at this recipe's LR that is a real but second-order
    change, and the checkpoint ladder + holdout will say whether 30k converged, which
    is the same way 70k was judged on the single-arm run.

    WALL-CLOCK, ESTIMATED AND NOT MEASURED: the single-arm recipe measured 2.6-3.5
    s/step on 2x A100 with two cameras. Same per-GPU batch, three cameras now, H100s
    roughly twice an A100 -- call it 1.5-2.5 s/step, so 30,000 steps is ~13-21 h,
    ~$150-250 at $11.79/hr. Measure the first 200 steps on the box and multiply
    before trusting either number.

PEAK LR IS DELIBERATELY NOT RESCALED
    Every prior FD run of this recipe is batch 64 at peak 3.5e-5. Linear/sqrt scaling
    lore says a bigger batch tolerates a bigger LR, but this run already changes
    embodiment, camera count, and data at once; changing the optimizer too would make
    any regression unattributable. Kept at 3.5e-5, which at batch 128 is effectively a
    HALVED per-sample LR -- conservative in the safe direction. If the loss curve is
    visibly lazier than `mf_pi05_ea`'s at matched sample counts, that is the first
    knob to revisit, as its own change.

WHY THE DRAW IS 64/64 AND NOT A TABLE OF POOLS
    Each half genuinely IS one v2.1 dataset on disk this time (`teleop_v21` /
    `ego_v21`), so there is no within-half split to compute -- the single-arm run's
    `_proportional_draws` machinery has nothing to do here. If a pool is ever added,
    split that half's 64 proportionally to frames, NOT equally, for the reasons
    documented in `piper/ego_vs_teleop.py`.

    The 50/50 is a hard per-batch count, so the gradient split is exact; and with the
    halves budget-matched to 3,182 frames (~2.3%), the revisit rates are near-equal
    too -- "the model saw more teleop" is (almost) not available as an explanation
    for anything, same as the single-arm run.

WHAT IS INHERITED AND SHOULD NOT MOVE
    peak LR 3.5e-5 -> 3.5e-6 cosine, LoRA `gemma_2b_lora` trunk with a full-rank
    action expert, EMA off, `action_dim=32` (mandatory padding -- see `pi05_arm`),
    `action_horizon=50`, `max_token_len=200`, augmentation on (both stacks, via the
    builder's single `augment` flag, default True). `action_horizon=50` is 2.5 s of
    future at Piper's 20 Hz, not comparable to YAM's 1.67 s.

    Early stop is the same stall guard as `mf_pi05_ea`, with the same caveats: expect
    it not to fire under cosine decay, and if it fires it leaves an un-annealed
    checkpoint. It decides when to stop paying; the holdout decides what to ship.

    Warmup is the same ABSOLUTE 1,000 steps (3.3% of this shorter run). The bound that
    matters is absolute -- fresh LoRA B-matrices produce their largest gradients in
    the first few hundred steps whatever the run length.

NUM_WORKERS=32 IS A GUESS WHERE 16 WAS A MEASUREMENT
    The measured 16 was two cameras, batch 64, 2 GPUs, and showed the run compute-
    bound. This run doubles samples per step, adds a third camera decode, and halves
    the model-step time per sample on H100s -- roughly 3x the decode demand per unit
    of compute time. 32 is the same per-GPU worker count as the measured setup times
    the new GPU count; A/B it against 16 in the first 200 steps like last time, and
    remember the vast.ai rule: size against the cgroup quota, not nproc.

RUNNING IT
    `run_yam.sh` derives its checkpoint subdirectory by stripping a known family
    prefix, which `mf_` does not match, so pass the experiment name explicitly:

        ./vast_run/run_yam.sh mf_pi05_dual_ea dual_ea
"""

from __future__ import annotations

from configs._shared.arms import pi05_arm
from configs._shared.robots import PIPER_DUAL
from configs._shared.schedule import Schedule
from configs.mf import datasets_dual as ds
from openpi.training import registry
import openpi.training.config as _config

BATCH_SIZE = 128
# The experiment. Half the gradient from each half of the corpus.
HALF_DRAW = BATCH_SIZE // 2

# `of_steps`: 30,000 is the thing being held fixed -- chosen to match the sample
# exposure of the originally-priced 60k @ 64 (3.84M samples), not to hit an epoch
# target. `describe()` reports the epochs each pool actually gets.
SCHEDULE = Schedule.of_steps(
    30_000,
    # Absolute, not the 2% default (which would give 600 -- SHORTER than anything
    # measured, the opposite problem the single-arm run's comment worried about).
    # Every FD run's peak grad norm has landed inside or just after the first 1,000.
    warmup_steps=1_000,
)

# Same stall/divergence guard as `mf_pi05_ea`, same parameters, same expectations.
EARLY_STOP = _config.EarlyStop(
    metric="loss",
    patience_steps=3_000,
    min_rel_delta=1e-3,
    min_steps=5_000,
)

SOURCES = (
    _config.MixtureSource(
        repo_id=ds.TELEOP_REPO,
        samples_per_batch=HALF_DRAW,
        root=ds.TELEOP_ROOT,
        # The contiguous 8-episode tail (78..85) -- 9.3% of a small-in-clips pool,
        # sized down from the single-arm 15. See the datasets module.
        exclude_episodes=ds.TELEOP_HOLDOUT_EPISODES,
    ),
    _config.MixtureSource(
        repo_id=ds.EGO_REPO,
        samples_per_batch=HALF_DRAW,
        root=ds.EGO_ROOT,
        # No holdout on this half, as before: the headline metric is the real rig, so
        # retargeted video is a training ingredient and never something scored.
    ),
)


registry.register(
    pi05_arm(
        "mf_pi05_dual_ea",
        robot=PIPER_DUAL,
        sources=SOURCES,
        # Its own asset_id, never shared: the norm stats describe THIS sampled 50/50
        # 14-D distribution and cannot be copied from the single-arm run or anywhere.
        asset_id="mf_dual_ea",
        schedule=SCHEDULE,
        batch_size=BATCH_SIZE,
        early_stop=EARLY_STOP,
        # Pinned so reordering the mixture cannot move where norm stats are read from.
        repo_id=ds.TELEOP_REPO,
        # See NUM_WORKERS above -- a guess to be A/B'd, not a measurement yet.
        num_workers=32,
        # Crash-resume granularity: ~half an hour of work at risk at ~2 s/step.
        save_interval=1_000,
        # THE LADDER IS THE POINT, as on the single-arm run: no in-training
        # validation, so shipping = scoring saved checkpoints against the teleop
        # holdout afterwards. `keep_period=5_000` pins 5k..25k = 5, plus the rolling 1
        # = ~78 GB at ~13 GB/checkpoint -- the same ladder density the 70k run got
        # from 10k pins. If disk is tight, 10_000 gives 2 pins (~39 GB) and a ladder
        # too coarse to find a knee; free space instead.
        max_to_keep=1,
        keep_period=5_000,
    )
)


def describe() -> str:
    """The epoch table. Reporting only -- nothing here feeds config.

    uv run python -c "from configs.mf.piper import dual_ego_vs_teleop as d; print(d.describe())"
    """
    per_source = (
        ("teleop (dual rig)", ds.TELEOP_TRAIN_FRAMES, HALF_DRAW),
        ("retargeted (ego)", ds.EGO_FRAMES, HALF_DRAW),
    )
    lines = [
        "mf_pi05_dual_ea",
        SCHEDULE.describe(per_source),
        "",
        f"teleop holdout: episodes {ds.TELEOP_HOLDOUT_EPISODES[0]}..{ds.TELEOP_HOLDOUT_EPISODES[-1]}",
        f"early stop: {EARLY_STOP}",
    ]
    if ds.TASK == "TBD":
        lines += ["", "NOTE: datasets_dual.TASK is still TBD -- read it from meta/tasks.jsonl."]
    return "\n".join(lines)

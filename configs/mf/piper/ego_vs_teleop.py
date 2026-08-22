"""mf / Piper: 50/50 retargeted human video and real teleop, on one plate-pick policy.

ONE arm. `mf_pi05_ea` -- pi0.5, half of every batch drawn from the real rig and half
from human video retargeted onto the same Piper:

    half        pools                             frames    duration    draw of 64
    ----------  --------------------------------  --------  ----------  ----------
    teleop      mrfood1 + mrfood3                  129,980   1 h 48.3 m     32
    retargeted  piper_ego + piper_stera_plate      130,115   1 h 48.4 m     32

There is no baseline arm here and no ablation. This is a single production-shaped run,
not a comparison, so nothing below is arranged to keep two arms matched -- which is the
one thing that would have to change if a second arm is ever added (see EARLY STOPPING).

WHY 50/50 MEANS THE SAME THING TWICE
    `samples_per_batch` is a hard per-batch count, so a source's gradient share is
    exactly `samples_per_batch / batch_size` regardless of how much of it sits on disk.
    Normally the storage ratio and the draw ratio are two different numbers and only the
    draw matters -- that is the most transferable result of the pi0.5 mixture study,
    where mix3070 (33% teleop on disk) and mix5050 (50% on disk) scored within noise
    because both drew a hard 32/32.

    Here they coincide: the pools were budget-matched at build time to 135 frames of
    each other, so 50/50 is true of the disk AND of the gradient. That is worth stating
    because it means this run has no revisit-rate asymmetry either -- each half is seen
    the same number of times, so "the model saw more teleop" is not available as an
    explanation for anything.

WHY THE FOUR POOLS ARE FOUR SOURCES AND NOT TWO
    They are four separate v2.1 datasets under `/workspace/final/`, so each needs its
    own `MixtureSource`. The draws WITHIN each half are frame-proportional, computed by
    `_proportional_draws` rather than typed in, which makes each half behave exactly as
    a single merged dataset of that half would: uniform sampling over a concatenation is
    the same thing as a per-source draw proportional to size.

    Proportional and not equal. Splitting the retargeted half 16/16 would oversample
    piper_stera_plate 5.4x relative to its size and quietly turn "1 h 48 m of retargeted
    video" into a stera-weighted mixture -- a second experimental variable nobody asked
    for. There is a real argument for boosting stera (it is the PLATE task, so it is the
    closer match to what mrfood does, while piper_ego is generic EgoDex pick-and-place),
    but that is a different experiment and it should be a second arm with its own
    `asset_id`, not an edit to this one.

WHY 70,000 STEPS
    The client's call, and the reasoning is undertraining: at batch 64 this is ~4.48M
    samples, ~17-18 passes over each pool (`describe()` prints the exact table). The
    previous generation of these runs -- the SO-101 pair at 2,400 then 5,000 steps --
    was demonstrably still descending when its cosine ran out, which makes any result a
    statement about the step count rather than about the data.

    BUDGET THIS AS A MULTI-DAY RUN. At the 2.6-3.5 s/step this recipe has measured on
    2x A100-80GB (two cameras here rather than YAM's three, so at the faster end),
    70,000 steps is roughly 50-68 hours; on 2x H100 nearer 33. ESTIMATED -- measure the
    first 200 steps on the actual box and multiply, before committing three days of
    rental to it.

    The cost of going this long is memorisation, and at ~18 epochs it is a real risk
    rather than a theoretical one. Against our own guidance -- 1-3 epochs of a scarce
    pool, memorisation risk past ~5 -- this is 3-6x over. Accepted knowingly: that band
    was calibrated on 250k-1M-frame multi-task pools where one epoch is already
    thousands of steps, and here one epoch of the whole ~253k-frame training corpus is
    3,959 steps. Steps (has the optimiser converged?) and epochs (am I memorising?) come apart
    at this scale, and obeying the band literally would mean a ~12k-step run.

    What makes that acceptable is the checkpoint ladder plus the teleop holdout, and
    only that pair. WATCH THE HOLDOUT, NOT THE TRAIN LOSS -- the train loss will keep
    falling all the way to 70k because the LR keeps falling, and it cannot distinguish
    fitting from memorising. A declining grad-norm trajectory is the same signature and
    is what flagged `pi05_ax91_mix9010`.

EARLY STOPPING, AND WHAT IT IS AND IS NOT FOR
    `EarlyStop(patience_steps=3_000)`: stop when the `log_interval` mean of `loss` has
    not improved by 0.1% for 3,000 consecutive steps. openpi has no such mechanism
    upstream; it was added for this run (`_config.EarlyStop`, `scripts/train.py`).

    EXPECT IT NOT TO FIRE. `Schedule.lr_schedule()` pins `decay_steps` to
    `num_train_steps`, so the LR is still falling at step 69,999, and a falling LR drags
    the training loss down with it. Under cosine decay a training-loss plateau is rare.
    This is a stall/divergence guard and a way to stop paying for a dead run -- not a
    convergence detector, and NOT a checkpoint-selection mechanism.

    IF IT DOES FIRE, WHAT IT LEAVES IS AN UN-ANNEALED CHECKPOINT. Stopping the cosine at
    step k of 70,000 leaves the model at step k's LR, which is not the model a k-step run
    would have produced -- that one would have annealed to 3.5e-6. So the stop decides
    when to stop spending; the holdout decides what to ship.

    `min_steps=5_000` keeps the window from opening inside warmup, where the loss moves
    for reasons unrelated to convergence. It is warmup (1,000) + patience (3,000) plus
    margin.

    AND IF A SECOND ARM IS EVER ADDED, THINK BEFORE COPYING THIS. Early stopping makes a
    config mean "at most N steps", so two arms that stop at different places are no
    longer matched on exposure OR on schedule position -- the exact double confound that
    made the four-arm pi0.5 mixture study unable to rank its own arms by loss. For a
    comparison, either turn it off on both or accept that only the holdout is readable.

BEFORE LAUNCHING: TWO PREFLIGHTS, BOTH CHEAP
    1. THE IMAGES ARE NOT READY. mrfood is 480x640 portrait rotated ~90 deg; the
       retargeted pools are 224x224 square upright. `resize_with_pad` preserves aspect,
       so half the batch would arrive letterboxed AND sideways relative to the other
       half. This is a conversion-time fix (rotate upright, centre-crop square, store
       224x224 h264) and NOT a training-time transform -- a transform the serving path
       does not also apply would show the policy a geometry on the real rig that it
       never trained on. Full argument in `configs/mf/datasets.py`.

    2. PER-DIM p1/p50/p99 OF `observation.state` AND `action`, teleop pool against
       retargeted pool, side by side. Thirty minutes; it has saved nine hours before.
       Quantile norm is computed over the BLEND, and the two halves are known to occupy
       the joint ranges differently -- the retargeted side saturates against every Piper
       limit (wrist joint5) while mrfood sits in a narrow interior band. If mrfood's real
       working range compresses into the middle of a range set by the saturated half,
       no learning rate fixes it. `pi05_ax91_mix9010` shipped exactly that.

       Dim 6 is the one to look at hardest even though it is already reconciled: both
       halves are binary {0,1}, so p1/p99 should read 0 and 1 on both. Anything else
       means a pool missed the binarisation.

NORM STATS ARE THIS ARM'S OWN
    `asset_id="mf_ea"`. They describe the SAMPLED 50/50 distribution, not any one pool,
    so they cannot be copied from anywhere and nothing can be copied from them. The
    fingerprint written beside them records the sources, roots, draws and holdout and is
    checked on load, so a `cp -r` from a neighbouring arm raises instead of training
    quietly against the wrong distribution.

WHAT IS INHERITED AND SHOULD NOT MOVE
    peak LR 3.5e-5 -> 3.5e-6 cosine, batch 64, LoRA `gemma_2b_lora` trunk with a
    full-rank action expert, EMA off, `action_dim=32`, `action_horizon=50`,
    `max_token_len=200`, augmentation on. All from `pi05_arm`, which documents each.

    `action_horizon=50` is 2.5 s of future here, because Piper runs at 20 Hz. The same
    50 buys 1.67 s on YAM and SO-101, so this arm's chunk is NOT comparable to theirs
    even though the number matches. The loader checks each pool's real fps against
    `PIPER_SINGLE.control_hz` and refuses a mixture whose sources disagree.

DELIBERATELY NOT HERE
    A `mf_pi05_base_control` arm -- `pi05_base` with a `gemma_2b` trunk and this arm's
    `asset_id`, for a zero-shot reference point in offline eval. It is five lines
    (see `configs/mm/so101/sim_vs_ego.py`) and costs no training. Add it when the eval
    runs, not before, since it is not part of this experiment.

RUNNING IT
    `run_yam.sh` derives its checkpoint subdirectory by stripping a known family prefix,
    which `mf_` does not match, so pass the experiment name explicitly:

        ./vast_run/run_yam.sh mf_pi05_ea ea
"""

from __future__ import annotations

from collections.abc import Sequence

from configs._shared.arms import pi05_arm
from configs._shared.robots import PIPER_SINGLE
from configs._shared.schedule import Schedule
from configs.mf import datasets as ds
from openpi.training import registry
import openpi.training.config as _config

BATCH_SIZE = 64
# The experiment. Half the gradient from each half of the corpus.
HALF_DRAW = BATCH_SIZE // 2

# `of_steps`, not `for_epochs`: 70,000 is the thing being held fixed, chosen against
# undertraining rather than against a data-exposure target. `describe()` reports the
# epochs each pool actually gets.
SCHEDULE = Schedule.of_steps(
    70_000,
    # 1.4% of the run. Absolute rather than the 2% default, and the same 1,000 the 50k
    # YAM run used, because the bound that matters here is absolute: the risky region is
    # the first few hundred steps whatever the run length, since freshly initialised
    # LoRA B-matrices and projections produce their largest gradients there. Every FD
    # run's peak grad norm has landed inside or just after warmup. The 2% default would
    # give 1,400, which is not wrong -- just longer than anything measured.
    warmup_steps=1_000,
)

EARLY_STOP = _config.EarlyStop(
    metric="loss",
    patience_steps=3_000,
    # 0.1% at some point in the window counts as still improving. Relative because loss
    # scale is arbitrary under per-arm norm stats -- an absolute epsilon tuned on one
    # mixture means nothing on the next.
    min_rel_delta=1e-3,
    # Past warmup (1,000) + patience (3,000), with margin.
    min_steps=5_000,
)


def _proportional_draws(frames: Sequence[int], total: int) -> tuple[int, ...]:
    """Split `total` samples across pools in proportion to their frame counts.

    Largest-remainder, so the result sums to `total` exactly -- which `pi05_arm` asserts
    anyway, but failing there would be a puzzle rather than a message.

    Computed rather than written down because the frame counts in `datasets.py` are the
    source of truth: change one and the draw follows, instead of a stale literal
    silently reweighting the mixture.
    """
    if total <= 0 or any(f <= 0 for f in frames):
        raise ValueError(f"_proportional_draws needs positive frames and total, got {frames} and {total}")
    exact = [total * f / sum(frames) for f in frames]
    draws = [int(x) for x in exact]
    # Hand out what flooring dropped, largest fractional part first.
    for i in sorted(range(len(exact)), key=lambda i: exact[i] - draws[i], reverse=True)[: total - sum(draws)]:
        draws[i] += 1
    return tuple(draws)


# --- teleop: the real rig, minus the eval holdout ---------------------------
# Proportional over TRAINED frames, not stored frames, since the holdout comes out of
# mrfood1 and mrfood3 unevenly.
_MRFOOD1_DRAW, _MRFOOD3_DRAW = _proportional_draws(
    (ds.MRFOOD1_TRAIN_FRAMES, ds.MRFOOD3_TRAIN_FRAMES), HALF_DRAW
)  # 15 / 17

# --- retargeted: consumed whole, nothing withheld ---------------------------
# No holdout on this half: the headline metric is the plate pick on the real rig, so
# retargeted video is a training ingredient and never something scored.
_EGO_DRAW, _STERA_DRAW = _proportional_draws((ds.PIPER_EGO_FRAMES, ds.PIPER_STERA_FRAMES), HALF_DRAW)  # 27 / 5

SOURCES = (
    _config.MixtureSource(
        repo_id=ds.MRFOOD1_REPO,
        samples_per_batch=_MRFOOD1_DRAW,
        root=ds.MRFOOD1_ROOT,
        exclude_episodes=ds.MRFOOD1_HOLDOUT_EPISODES,
    ),
    _config.MixtureSource(
        repo_id=ds.MRFOOD3_REPO,
        samples_per_batch=_MRFOOD3_DRAW,
        root=ds.MRFOOD3_ROOT,
        exclude_episodes=ds.MRFOOD3_HOLDOUT_EPISODES,
    ),
    _config.MixtureSource(
        repo_id=ds.PIPER_EGO_REPO,
        samples_per_batch=_EGO_DRAW,
        root=ds.PIPER_EGO_ROOT,
    ),
    _config.MixtureSource(
        repo_id=ds.PIPER_STERA_REPO,
        samples_per_batch=_STERA_DRAW,
        root=ds.PIPER_STERA_ROOT,
    ),
)


registry.register(
    pi05_arm(
        "mf_pi05_ea",
        robot=PIPER_SINGLE,
        sources=SOURCES,
        asset_id="mf_ea",
        schedule=SCHEDULE,
        batch_size=BATCH_SIZE,
        early_stop=EARLY_STOP,
        # `repo_id` names the PRIMARY source -- where `assets/<config>/<asset_id>` is
        # looked up. Pinned to mrfood1 rather than left to follow `sources[0]`, so
        # reordering the mixture cannot move where norm stats are read from.
        repo_id=ds.MRFOOD1_REPO,
        # Crash-resume granularity: ~50 min of work at risk at 3 s/step. `train.py`
        # always writes the final step on top of the interval, and now also writes
        # whatever step an early stop lands on.
        save_interval=1_000,
        # THE LADDER IS THE POINT, and it is what makes ~18 epochs survivable: openpi has
        # no in-training validation, so choosing what to ship means scoring saved
        # checkpoints against the teleop holdout after the fact. `max_to_keep=1` alone
        # would leave exactly one candidate and no way to tell 70k from its own knee.
        #
        # Disk, stated because it is the reason to change it: a pi0.5 checkpoint here is
        # ~13 GB (6.7 params + 5.9 optimizer state). `keep_period=10_000` pins 10k..60k
        # = 6, plus the rolling 1 = ~91 GB. If the box cannot hold that alongside the
        # datasets, `keep_period=20_000` gives 3 pins (~52 GB) and a coarser ladder.
        # Retention is a pure disk policy -- it never touches the optimizer, so changing
        # it does not change the run.
        max_to_keep=1,
        keep_period=10_000,
    )
)


def describe() -> str:
    """The epoch table. Reporting only -- nothing here feeds config.

        uv run python -c "from configs.mf.piper import ego_vs_teleop as e; print(e.describe())"

    The two mrfood rows are as good as `*_TRAIN_FRAMES`, which are estimated at each
    pool's mean episode length rather than summed from `meta/episodes.jsonl`.
    """
    per_source = (
        ("teleop/mrfood1", ds.MRFOOD1_TRAIN_FRAMES, _MRFOOD1_DRAW),
        ("teleop/mrfood3", ds.MRFOOD3_TRAIN_FRAMES, _MRFOOD3_DRAW),
        ("retgt/piper_ego", ds.PIPER_EGO_FRAMES, _EGO_DRAW),
        ("retgt/stera_plate", ds.PIPER_STERA_FRAMES, _STERA_DRAW),
    )
    halves = (
        ("TELEOP half", ds.TELEOP_TRAIN_FRAMES, HALF_DRAW),
        ("RETARGETED half", ds.EGO_FRAMES, HALF_DRAW),
    )
    return "\n".join(
        [
            "mf_pi05_ea",
            SCHEDULE.describe(per_source),
            "",
            "as two halves:",
            SCHEDULE.describe(halves).split("\n", 1)[1],
            "",
            f"early stop: {EARLY_STOP}",
        ]
    )

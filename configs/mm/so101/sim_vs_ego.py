"""mm / SO-101: does retargeted ego video buy anything on top of 9 minutes of sim?

Three pi0.5 arms, one task (*"pick up the cube and place it in the tray"*), one
embodiment, one holdout. They differ in what is in the batch and in nothing else.

    arm              total data     per-batch draw (of 64)    what it asks
    ---------------  -------------  ------------------------  --------------------------
    mm_pi05_sim10    ~9 min sim     sim 64                    the baseline
    mm_pi05_mix10    ~5 + ~5 min    sim 32 / ego 32           REPLACE half the sim with
                                                              ego at a fixed data budget
    mm_pi05_mix20    ~10 + ~10 min  sim 32 / ego 32           ADD ego on top, doubling
                                                              the data at fixed compute

`mm_pi05_sim10` vs `mm_pi05_mix10` is the clean A/B: same total minutes, same steps,
same schedule, same holdout -- composition is the only moving part. `mm_pi05_mix20`
then asks whether more data helps once the ratio is already balanced.

WHAT "5 MINUTES OF SIM" ACTUALLY MEANS HERE
    The client has 9 min 15 s of sim, not 10, so "10 minutes" is the whole pool and
    "5 minutes" is half of it. After the eval holdout comes out (10%), the full arms
    train on 8.33 min and the half arm on 4.31 min. Those are the real numbers, and they
    are what the epoch counts below are computed against.

WHY THE DRAW IS 32/32 AND NOT SOMETHING PROPORTIONAL
    `samples_per_batch` is a hard per-batch count, so a source's gradient share is
    exactly `samples_per_batch / batch_size` -- 50% here -- no matter how much of it sits
    on disk. This is the single most transferable result of the pi0.5 mixture study:
    mix3070 (33% teleop on disk) and mix5050 (50% on disk) scored within noise of each
    other because both drew a hard 32/32. Storage ratio sets the revisit rate; the draw
    sets what the optimiser balances. "5 min + 5 min" and "10 min + 10 min" are
    therefore statements about the POOLS, and both arms sit at the same 50/50 gradient
    share by construction.

WHY ALL THREE RUN THE SAME NUMBER OF STEPS
    Unequal step counts are a double confound: they change exposure AND schedule
    position, because `decay_steps` tracks `num_train_steps`, so a longer run spends
    more absolute steps at floor LR and reports a flattered final loss. The pi0.5
    mixture study baked this in (11.4k/17.7k/23.6k/32k) and could not then use training
    loss to rank its arms at all. One `_SCHEDULE` object is shared by all three arms
    here so they cannot drift apart.

WHAT 2,400 STEPS BUYS, AND THE EPOCH SPREAD IT LEAVES
    Roughly 154k samples per arm. The epochs each pool gets fall out of that and differ
    per arm by design, because the arms differ in pool size:

        mm_pi05_sim10   sim 10.25
        mm_pi05_mix10   sim/half 9.90   ego 8.44
        mm_pi05_mix20   sim 5.12        ego 4.22

    Note `sim10` at 10.25 and `mix10`'s sim half at 9.90 are nearly equal -- half the
    draw on half the data. So those two arms give the sim pool the same number of
    passes, and the only difference is that half of `mix10`'s gradient comes from ego
    instead of from more sim. That is what makes the pair a clean A/B rather than two
    runs that happen to have the same length.

    Against our own guidance -- 1-3 epochs of a scarce pool, memorisation risk past
    ~5 -- `mix20` sits just at the edge and the other two are about 2x over it. That is
    a real cost accepted knowingly, and it is not fixable by shortening the run: that
    guidance was calibrated on 250k-1M-frame multi-task pools where one epoch is already
    thousands of steps, whereas here one epoch of the ENTIRE dataset is 234 steps. The
    two things the rule conflates -- steps (has the optimiser converged?) and epochs (am
    I memorising?) -- come apart completely at nine minutes of data, and obeying the
    epoch band literally would mean a ~1,200-step run that has barely left warmup.

    So the design moves the stopping decision downstream instead: save every 300 steps
    and let the holdout choose the checkpoint. That is why 10% of a very small pool is
    spent on `SIM_HOLDOUT_EPISODES`.

    What to actually look at, in order: holdout MSE per checkpoint (if it stalls or
    worsens while train loss keeps falling, ship the earlier checkpoint -- that is the
    signal this design exists to catch); grad-norm trajectory (a DECLINING one is the
    memorisation signature that flagged mix9010); and `chunk_last` rather than aggregate
    loss, since delta actions make t=0 nearly free.

    If the holdout says every arm peaked early, the fix is fewer steps and a re-run of
    all three, never a shorter run of one -- they have to stay matched. The cosine is
    spent at the end of a run, so a finished arm cannot usefully be extended either.

BEFORE LAUNCHING, DO THE NORM-STATS PREFLIGHT (30 minutes, saves 9 hours)
    Per-dim p1/p50/p99 of `observation.state` and `action`, for the sim pool and the ego
    pool side by side. Any channel whose ranges do not overlap sanely is a bug, and the
    gripper is where it will be: quantile norm is computed over the blended mixture, so
    a pool whose gripper lives on a different scale gets crushed under the other's range
    and flow matching averages the two into a half-open gripper. This is exactly what
    `pi05_ax91_mix9010` shipped, and no learning rate fixes it. Item 3 of
    `configs/mm/datasets.py` lists the other three schema traps -- joint ORDER is the
    one most likely to bite here, since stage 6 emits the SO-101 columns reversed.

NORM STATS ARE PER-ARM AND MUST NOT BE COPIED BETWEEN THESE THREE
    Three different mixtures over overlapping data means three different sampled
    distributions, hence `mm_sim10` / `mm_mix10` / `mm_mix20`. The fingerprint written
    beside the stats records the sources, roots, draws and holdout fractions and is
    checked on load, so a `cp -r` from a neighbouring arm now raises instead of training
    quietly against the wrong distribution.

WHAT IS INHERITED AND SHOULD NOT MOVE
    peak LR 3.5e-5 -> 3.5e-6 cosine, batch 64, LoRA `gemma_2b_lora` trunk with a
    full-rank action expert, EMA off, `action_dim=32`, `action_horizon=50`,
    `max_token_len=200`, augmentation on. All of it comes from `pi05_arm`, which
    documents why each is what it is. Changing any of them at the same time as the data
    composition would make the result unattributable -- and composition is the
    experiment.

    Note `action_horizon=50` is 1.67 s here, the same as on YAM, because SO-101 is also
    30 Hz. The loader verifies that against each dataset's real fps and refuses a
    mixture whose sources disagree.

RUNNING THEM
    `run_yam.sh` derives its checkpoint subdirectory by stripping a known family prefix,
    which these names do not match, so pass the experiment name explicitly:

        ./vast_run/run_yam.sh mm_pi05_sim10 sim10
        ./vast_run/run_yam.sh mm_pi05_mix10 mix10
        ./vast_run/run_yam.sh mm_pi05_mix20 mix20

    `mm_pi05_sim10` is runnable as soon as the sim dataset is converted to v2.1; the two
    mixture arms also need the retargeted ego dataset, which does not exist yet. Run the
    baseline first -- it is the one the other two are measured against.
"""

from configs._shared.arms import pi05_arm
from configs._shared.robots import SO101
from configs._shared.schedule import Schedule
from configs.mm import datasets as ds
from openpi.training import registry
import openpi.training.config as _config

BATCH_SIZE = 64
# Half the batch from each source on the mixture arms. Gradient share is exactly this
# over BATCH_SIZE, independent of pool sizes.
_MIX_DRAW = BATCH_SIZE // 2

# Shared by all three arms so their step counts and schedule positions cannot diverge.
#
# `of_steps` rather than `for_epochs` because the run LENGTH is the thing being held
# fixed here, not data exposure -- the three arms have different pool sizes, so no single
# epoch target could describe all of them anyway. `describe()` reports the epochs each
# arm actually gets.
_SCHEDULE = Schedule.of_steps(
    2_400,
    # 10.4% of the run, which is high against the usual 2-5% -- kept anyway because the
    # bound that matters here is ABSOLUTE, not fractional. The risky region is the first
    # few hundred steps whatever the run length: freshly initialised LoRA B-matrices and
    # projections produce their largest gradients there, and every FD run's peak grad
    # norm has landed inside or just after warmup. Scaling this to 5% would put it at
    # ~120, below the point where our own runs stopped spiking.
    warmup_steps=250,
)


def _sim_source(samples_per_batch: int, *, half: bool) -> _config.MixtureSource:
    """The sim pool, minus the eval holdout -- and minus half the rest when `half`.

    `SIM_HALF_EXCLUDE_EPISODES` already contains the holdout, so the two cases differ in
    which list is passed and never in whether the holdout is applied.
    """
    return _config.MixtureSource(
        repo_id=ds.SIM_REPO,
        samples_per_batch=samples_per_batch,
        root=ds.SIM_ROOT,
        exclude_episodes=ds.SIM_HALF_EXCLUDE_EPISODES if half else ds.SIM_HOLDOUT_EPISODES,
    )


def _ego_source(samples_per_batch: int, *, holdout_fraction: float) -> _config.MixtureSource:
    """The ego pool, subsampled to the minutes this arm is meant to have.

    No eval holdout on the ego side: the headline metric is sim-task performance on sim
    frames, so ego is a training ingredient here and never something we score.
    """
    return _config.MixtureSource(
        repo_id=ds.EGO_REPO,
        samples_per_batch=samples_per_batch,
        root=ds.EGO_ROOT,
        holdout_fraction=holdout_fraction,
        holdout_seed=ds.EGO_HOLDOUT_SEED,
    )


# --- 1. baseline: all the sim there is, nothing else ------------------------
SIM10_SOURCES = (_sim_source(BATCH_SIZE, half=False),)

# --- 2. same budget, half of it ego ----------------------------------------
MIX10_SOURCES = (
    _sim_source(_MIX_DRAW, half=True),
    _ego_source(_MIX_DRAW, holdout_fraction=ds.EGO_QUARTER_HOLDOUT_FRACTION),
)

# --- 3. double the budget, still balanced ----------------------------------
MIX20_SOURCES = (
    _sim_source(_MIX_DRAW, half=False),
    _ego_source(_MIX_DRAW, holdout_fraction=ds.EGO_HALF_HOLDOUT_FRACTION),
)


registry.register(
    pi05_arm(
        "mm_pi05_sim10",
        robot=SO101,
        sources=SIM10_SOURCES,
        asset_id="mm_sim10",
        schedule=_SCHEDULE,
        batch_size=BATCH_SIZE,
        # 8 checkpoints written, 6 kept: the rolling window holds 1500-2400 and
        # `keep_period` pins 600 and 1200 on top of it, so the curve is sampled across
        # the WHOLE run and not just its tail. That matters because the whole design
        # leans on picking a checkpoint by holdout score rather than taking the last
        # one, and at ~10 epochs the knee can easily land in the first half. Six pi0.5
        # checkpoints is the disk cost of being able to see it.
        save_interval=300,
        max_to_keep=4,
        keep_period=600,
    ),
    pi05_arm(
        "mm_pi05_mix10",
        robot=SO101,
        sources=MIX10_SOURCES,
        asset_id="mm_mix10",
        schedule=_SCHEDULE,
        batch_size=BATCH_SIZE,
        # `repo_id` names the PRIMARY source, which is where `assets/<config>/<asset_id>`
        # is looked up. Pinned to sim on both mixture arms so all three arms agree, and
        # so it does not silently follow `sources[0]`.
        repo_id=ds.SIM_REPO,
        save_interval=300,
        max_to_keep=4,
        keep_period=600,
    ),
    pi05_arm(
        "mm_pi05_mix20",
        robot=SO101,
        sources=MIX20_SOURCES,
        asset_id="mm_mix20",
        schedule=_SCHEDULE,
        batch_size=BATCH_SIZE,
        repo_id=ds.SIM_REPO,
        save_interval=300,
        max_to_keep=4,
        keep_period=600,
    ),
)


def describe() -> str:
    """The epoch table for all three arms. Reporting only -- nothing here feeds config.

        uv run python -c "from configs.mm.so101 import sim_vs_ego; print(sim_vs_ego.describe())"

    Ego rows are as good as `EGO_TOTAL_FRAMES`, which is still an estimate.
    """
    arms = (
        ("mm_pi05_sim10", (("sim", ds.SIM_TRAIN_FRAMES, BATCH_SIZE),)),
        (
            "mm_pi05_mix10",
            (("sim/half", ds.SIM_HALF_TRAIN_FRAMES, _MIX_DRAW), ("ego/5min", ds.EGO_QUARTER_TRAIN_FRAMES, _MIX_DRAW)),
        ),
        (
            "mm_pi05_mix20",
            (("sim", ds.SIM_TRAIN_FRAMES, _MIX_DRAW), ("ego/10min", ds.EGO_HALF_TRAIN_FRAMES, _MIX_DRAW)),
        ),
    )
    return "\n".join(f"{name}\n{_SCHEDULE.describe(sources)}" for name, sources in arms)

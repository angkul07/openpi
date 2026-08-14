"""The 7h mixture arms: the same experiment under pi0-FAST and under pi0.5.

Same two upstreams as the 15h arms, pre-selected down to 7.014 h and stored locally as
standalone datasets renumbered 0..N-1:

    teleop  251,703 frames (2.331 h)   ego  505,837 frames (4.684 h)

Storage is 33/67 here, but the fixed per-batch draw means the gradient share is set by
sampling regardless -- never by how much is on disk.

WHY THE STEP COUNTS ARE WHAT THEY ARE
    They come from holding the ORIGINAL epoch budget, not from scaling by the hours
    removed. The 15h E-A ran 3.0 teleop epochs; keeping that on 7h gives the E-A/E-B
    length below, which also lands ego at almost exactly 1.5. That derivation is now
    computed by `Schedule.for_epochs` instead of being asserted in a comment.

    E-B holds the SAME length rather than recomputing from its 40/24 split: the arms
    must differ in mixing ratio alone, so unequal optimisation budgets would confound
    the comparison. E-C stays the long arm at 2x E-A.

WHY THESE NEED THEIR OWN NORM STATS (asset_id yam7h_*, not yam_mix_*)
    The mixture ratio is unchanged from the 15h arms but the underlying distribution is
    not: ego dropped 160 low-count objects and teleop is a different 761-episode sample,
    so the q01/q99 quantiles move. Reusing yam_mix_p50 would silently normalise against
    the 15h distribution -- and `run_yam.sh` skips stat computation whenever the file
    already exists. (Since this refactor a fingerprint is written alongside the stats
    and checked on load, so that mistake now raises instead of training quietly.)

WHAT IS DELIBERATELY UNCHANGED from the 15h arms
    peak LR 3.5e-5, batch 64, LoRA variant, EMA disabled. The previous run was
    undertrained, not misoptimised -- grad norms were stable -- so only
    schedule-length-dependent knobs move. Changing LR or batch size at the same time as
    dataset scale would make the result unattributable.

    These are fresh runs from the base checkpoints, NOT resumes of the 7k checkpoint
    (a resume would drag along its exhausted cosine schedule and old norm stats).

THE pi05 TWINS
    `pi05_yam7h_*` uses a data config that is identical to its `pi0_fast_yam7h_*`
    counterpart, so ARCHITECTURE IS THE ONLY MOVING PART. The family deltas
    (action_dim 32, max_token_len 200, discrete_state_input left unset, num_workers 16)
    all live in `pi05_arm`, documented there.

    Norm stats are REUSED from the pi0-FAST arms -- do not recompute.
    `compute_norm_stats.py` applies repack + data_transforms only (never
    model_transforms), and `use_quantile_norm` is `model_type != PI0`, true for
    PI0_FAST and PI05 alike, so the 14-dim quantile stats are identical:
        cp -r assets/pi0_fast_yam7h_ea/yam7h_p50  assets/pi05_yam7h_ea/
        cp -r assets/pi0_fast_yam7h_eb/yam7h_p625 assets/pi05_yam7h_eb/
        cp -r assets/pi0_fast_yam7h_ec/yam7h_p50  assets/pi05_yam7h_ec/
    run_yam.sh stage [1/3] then skips recomputation on its own. This is the one case
    where copying stats between arms is CORRECT; the fingerprint travels with them and
    still matches, because the mixture is genuinely the same.

    pi0.5 checkpoints are ~25-30% larger than pi0-FAST (+311M expert params and their
    optimizer state). Drop max_to_keep to 2 if the checkpoint volume is tight.
"""

from configs._shared.arms import pi0_fast_arm
from configs._shared.arms import pi05_arm
from configs._shared.schedule import Schedule
from configs.fd import datasets as ds
from openpi.training import registry
import openpi.training.config as _config

# E-A/E-B length: 3.0 teleop epochs at the baseline 32/64 draw, rounded to a round 100.
# Ego lands at ~1.49 epochs as a consequence, not by separate choice.
_BASE = Schedule.for_epochs(
    frames=ds.TELEOP_7H_FRAMES,
    batch_size=32,  # teleop's per-batch draw on the baseline arm, not the training batch
    epochs=3.0,
    round_to=100,
    warmup_steps=500,
)
# E-C: twice the length, warmup ~2% of it.
_LONG = Schedule.of_steps(2 * _BASE.num_train_steps, warmup_steps=950)


def _sources(teleop_per_batch: int, batch_size: int = 64) -> tuple[_config.MixtureSource, ...]:
    return (
        _config.MixtureSource(
            repo_id=ds.TELEOP_REPO,
            samples_per_batch=teleop_per_batch,
            root=ds.TELEOP_7H_ROOT,
            # MUST stay empty. This root is a renumbered 0..760 subset that already
            # physically excludes the holdout, while TELEOP_HOLDOUT_EPISODES holds
            # ORIGINAL abc-teleop indices -- wrong twice over here: 143 of the 249 fall
            # outside 0..760 (which raises), and the other 106 are in range and would
            # silently withhold completely unrelated episodes.
            exclude_episodes=(),
        ),
        _config.MixtureSource(
            repo_id=ds.EGO_REPO,
            samples_per_batch=batch_size - teleop_per_batch,
            root=ds.EGO_7H_ROOT,
            # No ego holdout: headline metrics are teleop-only by design.
        ),
    )


# (suffix, asset_id, teleop_per_batch, schedule)
_ARMS = [
    # E-A: baseline ratio.
    ("ea", "yam7h_p50", 32, _BASE),
    # E-B: teleop-biased (62.5% gradient share).
    ("eb", "yam7h_p625", 40, _BASE),
    # E-C: 2x length at the baseline ratio.
    ("ec", "yam7h_p50", 32, _LONG),
]

for _suffix, _asset_id, _teleop_per_batch, _schedule in _ARMS:
    _srcs = _sources(_teleop_per_batch)
    registry.register(
        pi0_fast_arm(
            f"pi0_fast_yam7h_{_suffix}",
            sources=_srcs,
            asset_id=_asset_id,
            schedule=_schedule,
            repo_id=ds.TELEOP_REPO,
            keep_period=None,
        ),
        pi05_arm(
            f"pi05_yam7h_{_suffix}",
            sources=_srcs,
            asset_id=_asset_id,
            schedule=_schedule,
            repo_id=ds.TELEOP_REPO,
            keep_period=None,
        ),
    )

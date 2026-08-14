"""The 15h teleop-oversampling arms (E-A / E-B / E-C), pi0-FAST.

The original mixture experiment. Two full source datasets, stored at ~30% teleop /
70% ego by frames once the holdout is withheld, sampled at a FIXED per-batch count so
teleop's gradient share is set explicitly rather than inherited from storage:

    teleop  angkul07/abc-teleop                              460,647 frames  (4.265 h)
    ego     angkul07/EgoDex-PickPlace-YAM-14dof-multiview  1,074,893 frames  (9.953 h)

There is deliberately no per-source loss weighting -- sampling is the single lever.

These arms read the FULL datasets from `$HF_LEROBOT_HOME/<repo_id>` (root=None) and
withhold the holdout BY INDEX. That is the one place `TELEOP_HOLDOUT_EPISODES` is
correct to use: every later arm runs on a pre-selected subset that is renumbered
0..N-1 and already excludes those episodes physically.

Epoch counts are reported by `Schedule.describe()` rather than asserted in a comment.
The three arms exist to separate two things: E-A vs E-B is MIXING RATIO at a fixed
budget, and E-A vs E-C is BUDGET at a fixed ratio.
"""

from configs._shared.arms import pi0_fast_arm
from configs._shared.schedule import Schedule
from configs.fd import datasets as ds
from openpi.training import registry
import openpi.training.config as _config

# 50,000 steps is the official-recipe length, not a derived number. E-A and E-B hold
# it identical on purpose: the two arms must differ in MIXING RATIO alone, so giving
# them unequal optimisation budgets would confound the comparison. E-C is the long arm.
_STEPS = 50_000
_WARMUP = 1_000


def _sources(teleop_per_batch: int, batch_size: int = 64) -> tuple[_config.MixtureSource, ...]:
    return (
        _config.MixtureSource(
            repo_id=ds.TELEOP_REPO,
            samples_per_batch=teleop_per_batch,
            # None -> resolves $HF_LEROBOT_HOME/<repo_id>. These are the full datasets.
            root=None,
            # Withheld for offline eval; also what makes training storage 30/70.
            # Nothing moves on disk -- these episodes are simply never sampled.
            exclude_episodes=ds.TELEOP_HOLDOUT_EPISODES,
        ),
        _config.MixtureSource(
            repo_id=ds.EGO_REPO,
            samples_per_batch=batch_size - teleop_per_batch,
            root=None,
            # No ego holdout: headline metrics are teleop-only by design.
        ),
    )


_ARMS = [
    # E-A: baseline ratio, matched to the previous 50/50 experiment.
    ("pi0_fast_yam_mix_ea", "yam_mix_p50", 32, Schedule.of_steps(_STEPS, warmup_steps=_WARMUP)),
    # E-B: teleop-biased arm (62.5% gradient share on the higher-quality source).
    ("pi0_fast_yam_mix_eb", "yam_mix_p625", 40, Schedule.of_steps(_STEPS, warmup_steps=_WARMUP)),
    # E-C: long run at the safe ratio, aligned with the official recipe length.
    ("pi0_fast_yam_mix_ec", "yam_mix_p50", 32, Schedule.of_steps(2 * _STEPS, warmup_steps=2 * _WARMUP)),
]

for _name, _asset_id, _teleop_per_batch, _schedule in _ARMS:
    registry.register(
        pi0_fast_arm(
            _name,
            sources=_sources(_teleop_per_batch),
            # Per-ratio norm stats. p50 and p625 draw different distributions through
            # the same sampler, so they cannot share an asset id.
            asset_id=_asset_id,
            schedule=_schedule,
            repo_id=ds.TELEOP_REPO,
            keep_period=None,
        )
    )

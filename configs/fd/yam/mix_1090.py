"""pi05_yam1090_ea -- the 10/90 arm: same 7.014 h, teleop cut to 10% of the frames.

Fourth point on the mixture-ratio sweep. Storage teleop share across the series:

    100% (pi05_abcego_sd) -> 50% (pi05_50run_ea) -> 33% (pi05_yam7h_ea) -> 10% (here)

Total hours are held at 7.014 h throughout; only the split moves.

    teleop   75,754 frames (0.701 h,  ~229 eps)   10.0% of stored frames
    ego     681,786 frames (6.313 h, ~5,626 eps)  90.0%

The 24/40 draw gives teleop a 37.5% gradient share against a 10% storage share -- a
3.75x oversample -- and lands teleop at ~10.1 epochs against ego's ~1.9. Call
`SCHEDULE.describe(...)` (below) for the exact table rather than trusting these.

WHY 32,000 STEPS AND NOT 23,600
    The "hold the step count fixed" rule that ties E-A and E-B together exists to keep
    two arms of ONE experiment differing in ratio alone. This is a different mixture.
    At 40 ego samples per batch, 23,600 steps would leave ego at 1.39 epochs; 32,000
    puts it at ~1.88, close to yam7h_ea's ~1.49, so ego exposure stays roughly
    comparable across the two mixtures and TELEOP VOLUME is the variable.

WHY keep_period IS SET (the yam7h arms use None)
    At ~10 teleop epochs, overfitting is the failure mode to watch, and max_to_keep=4
    with save_interval=1,000 retains only a 4,000-step rolling window -- every early
    checkpoint would be gone before it could be scored. Pinning 5k/10k/.../30k lets the
    overfit knee be located after the fact. Set to None if checkpoint volume is tight
    (each is ~13 GB).

AUGMENTATION IS ON, inherited unchanged from the yam7h arms. Worth knowing before
reading the loss curve: it augments IMAGES ONLY. The state and action streams repeat
verbatim on all ~10 teleop revisits, so it blunts visual memorisation, not action
memorisation -- which is the other reason keep_period is set.

FRESH NORM STATS (asset_id yam1090_p375). Do not copy yam7h_* or reuse p50: both the
ratio and the underlying distribution moved -- teleop is a ~229-episode subsample and
ego gained ~176k frames from objects the 30-episode-minimum filter had excluded -- so
the q01/q99 quantiles shift on both sides.

Cost: ~12.3 h on 2x H100 SXM, extrapolated from the 1.388 s/step measured on
pi05_50run_ea at the same batch size, architecture and camera count.
"""

from configs._shared.arms import pi05_arm
from configs._shared.robots import YAM
from configs._shared.schedule import Schedule
from configs.fd import datasets as ds
from openpi.training import registry
import openpi.training.config as _config

_TELEOP_PER_BATCH = 24
_EGO_PER_BATCH = 40

# Chosen to land ego near the yam7h arms' exposure; warmup is 2.19% of it, matching the
# ~2.1% used throughout.
SCHEDULE = Schedule.of_steps(32_000, warmup_steps=700)

SOURCES = (
    _config.MixtureSource(
        repo_id=ds.TELEOP_REPO,
        samples_per_batch=_TELEOP_PER_BATCH,
        root=ds.TELEOP_1090_ROOT,
        # Empty for the same reason as the 7h arms: this root is a renumbered 0..228
        # subset that already physically excludes the holdout, while
        # TELEOP_HOLDOUT_EPISODES holds ORIGINAL abc-teleop indices.
        exclude_episodes=(),
    ),
    _config.MixtureSource(
        repo_id=ds.EGO_REPO,
        samples_per_batch=_EGO_PER_BATCH,
        root=ds.EGO_1090_ROOT,
        # No ego holdout: headline metrics are teleop-only by design.
    ),
)

# The epoch table this arm's comment block used to assert by hand:
#   print(SCHEDULE.describe([
#       ("teleop", ds.TELEOP_1090_FRAMES, _TELEOP_PER_BATCH),
#       ("ego", ds.EGO_1090_FRAMES, _EGO_PER_BATCH),
#   ]))

registry.register(
    pi05_arm(
        "pi05_yam1090_ea",
        robot=YAM,
        sources=SOURCES,
        asset_id="yam1090_p375",
        schedule=SCHEDULE,
        repo_id=ds.TELEOP_REPO,
        keep_period=5_000,
    )
)

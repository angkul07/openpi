"""<CLIENT> <EXPERIMENT> -- one line saying what this experiment is trying to separate.

Copy this file per experiment. The docstring is the point of the file: put the
REASONING here, next to the arm, where it can be read by whoever inherits the run.

What belongs in here:

  * What the sources are, with measured frame counts and hours.
  * What the independent variable is, and what is deliberately held fixed so the
    result stays attributable.
  * Why the step count is what it is -- derived from an epoch target, or copied from
    another run for comparability. Say which.
  * Which mistakes are pre-empted: index spaces, stats that must not be copied,
    anything that fails silently rather than loudly.

What does NOT belong in here: arithmetic you can compute. Use `Schedule.for_epochs`
and `Schedule.describe()` instead of a hand-written epoch table that goes stale the
first time someone edits `batch_size`.
"""

from configs._shared.arms import pi05_arm
from configs._shared.schedule import Schedule
from configs._template import datasets as ds  # -> `from configs.<client> import datasets as ds`
from openpi.training import registry
import openpi.training.config as _config

_BATCH_SIZE = 64
_TELEOP_PER_BATCH = 32

# Sized to a data-exposure target, so changing batch_size recomputes it rather than
# silently changing the epoch count. Use `Schedule.of_steps(...)` instead when the step
# count is the thing being held fixed (e.g. matching another arm for comparability).
SCHEDULE = Schedule.for_epochs(
    frames=ds.TELEOP_FRAMES,
    batch_size=_TELEOP_PER_BATCH,
    epochs=3.0,
    round_to=100,
    warmup_steps=500,
)

SOURCES = (
    _config.MixtureSource(
        repo_id=ds.TELEOP_REPO,
        samples_per_batch=_TELEOP_PER_BATCH,
        root=ds.TELEOP_ROOT,
        exclude_episodes=ds.HOLDOUT_EPISODES,
    ),
    _config.MixtureSource(
        repo_id=ds.EGO_REPO,
        samples_per_batch=_BATCH_SIZE - _TELEOP_PER_BATCH,
        root=ds.EGO_ROOT,
    ),
)

# Uncomment to print the epoch table this experiment would otherwise assert by hand:
#   print(SCHEDULE.describe([
#       ("teleop", ds.TELEOP_FRAMES, _TELEOP_PER_BATCH),
#       ("ego", ds.EGO_FRAMES, _BATCH_SIZE - _TELEOP_PER_BATCH),
#   ]))

# NOTE the name is prefixed with the client slug. Config names are GLOBAL -- an
# unprefixed name will eventually collide with another client's or a built-in openpi
# config, and registration raises when it does.
registry.register(
    pi05_arm(
        "client_pi05_ea",
        sources=SOURCES,
        # Per-mixture norm stats. Never share an asset_id between two different draws
        # or two different underlying distributions.
        asset_id="client_p50",
        schedule=SCHEDULE,
        batch_size=_BATCH_SIZE,
    )
)

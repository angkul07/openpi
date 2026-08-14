"""pi05_50run_ea -- the 50/50 ego+teleop merge, from ONE pre-merged dataset.

The openpi counterpart of the LeRobot run in `fd/sdk/lerobot_run`
(`run_yam_lerobot.sh pi05_50_ea`). Same data, same schedule, same trainable set -- so
the two are directly comparable, which is the point of having both.

Source: `angkul07/50_run_v21_fixed`, LeRobot v2.1.
    4,272 episodes / 755,964 frames / 7.00 h at 30 fps, 3 cameras at 224x224.
    3.5 h ego + 3.5 h teleop merged into ONE dataset at 50.00%/50.00% by frames.
    The teleop half excludes every episode in abc-teleop-holdout (249/249 matched by
    content hash, 0 leaks), so offline eval on that holdout is honest.

THE RATIO LIVES IN STORAGE, NOT IN THE SAMPLER
    The yam7h arms hold two roots and draw a hard 32/32 every batch. Here the 50/50 is
    already baked into one dataset, so a single source drawing all 64 gives the same
    expected ratio -- but 50/50 IN EXPECTATION, Binomial(64, 0.5), sd ~4 samples/batch,
    rather than exact per batch. Fine in aggregate over 17.7k steps, and it is
    precisely the constraint that forced the LeRobot port to merge in the first place,
    since LeRobot has no per-source count. Keeping openpi on the merged dataset is what
    makes the two runs comparable; use the yam7h arms if you want exact batches.

    Still a mixture-of-one for the same two mechanical reasons as `pi05_abcego_sd`
    (root=None off the mixture path, and run_yam.sh stage [0]) -- see that module.

NO AUGMENTATION, both stacks off, matching the LeRobot run which applied none.
    Leaving either on would break comparability with the LeRobot numbers.

use_delta_joint_actions STAYS TRUE (inherited default), and the mask is correct here
    for a non-obvious reason: make_bool_mask(6, -1, 6, -1) is 6 joints delta + 1
    gripper absolute PER ARM, and this dataset is ordered [R j1-6, R grip, L j1-6,
    L grip] -- right arm FIRST, unlike the abc-ego set's left-first layout. The mask is
    symmetric across the two arms, so it is correct either way; only a mask with
    different per-arm structure would care. This is openpi's positional equivalent of
    the LeRobot side's `relative_exclude_joints=['gripper']`, which resolves BY NAME.

NORM STATS MUST BE COMPUTED, not copied from yam7h_* or abcego_*. Quantile stats are
    computed AFTER data_transforms, i.e. in DELTA space -- the same space the LeRobot
    run normalised in, so the two are on equal footing.

17,700 STEPS IS NOT A ROUND EPOCH, deliberately. One epoch is 755,964 / 64 = 11,812
    steps, so this is 1.4985 epochs. Carried over VERBATIM from the LeRobot run so the
    two match; that is why it is spelled as a literal here rather than derived from an
    epoch target. Use `Schedule.for_epochs(..., epochs=1.5)` if you ever want exactly
    1.5 (17,718) and no longer need to match.

batch_size 64 with --fsdp-devices 1 = pure data parallel, 32/GPU. Measured on
2x RTX PRO 6000 Blackwell (96 GB) under the PyTorch LeRobot port: 32.6 GB/GPU at
32/GPU, so 80 GB-class hardware clears this comfortably. Note openpi is JAX/XLA here
versus eager PyTorch there, so step time will NOT match the 4.49 s/step measured on the
LeRobot side; only the recipe is shared.
"""

from configs._shared.arms import pi05_arm
from configs._shared.robots import YAM
from configs._shared.schedule import Schedule
from configs.fd import datasets as ds
from openpi.training import registry
import openpi.training.config as _config

_BATCH_SIZE = 64

# Literal, not derived -- it is copied from the LeRobot run so the two are comparable.
# warmup 350 is 2.0% of the run, matching the yam7h arms' 500/23,600.
SCHEDULE = Schedule.of_steps(17_700, warmup_steps=350)

registry.register(
    pi05_arm(
        "pi05_50run_ea",
        robot=YAM,
        sources=(
            _config.MixtureSource(
                repo_id=ds.RUN50_REPO,
                samples_per_batch=_BATCH_SIZE,  # == batch_size: the only source
                root=ds.RUN50_ROOT,
                # Holdout is already physically absent from this dataset (excluded at
                # build time), so there is nothing to withhold here.
                exclude_episodes=(),
            ),
        ),
        asset_id="yam50run",
        schedule=SCHEDULE,
        batch_size=_BATCH_SIZE,
        augment=False,
        # 1,500 matches the LeRobot run's save_freq. 11 saves over 17.7k steps; at
        # ~15-25 GB each that is 165-275 GB if you keep them all, which is why
        # max_to_keep is set. Bump it only if the box has the disk.
        save_interval=1_500,
        keep_period=5_000,
    )
)

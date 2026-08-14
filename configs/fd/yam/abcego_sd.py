"""pi05_abcego_sd -- 100% teleop, single source, exactly one epoch, no augmentation.

Source: `angkul07/abc-ego` `put_the_screwdriver_in_the_bin`, converted from MCAP by
`vast_run/mcap_to_lerobot.py`. 2,234 episodes / 730,496 frames / 6.76 h at 30 fps.
Single task string, so `prompt` carries no discriminative signal -- expected for a
single-task finetune, but it does mean language is doing nothing here.

A MIXTURE OF ONE, DELIBERATELY
    100% teleop is one source, so the plain single-source `LeRobotYamDataConfig` looks
    like the obvious choice. It is the wrong one, for two mechanical reasons:
      1. `create_torch_dataset()` hardcodes root=None on the non-mixture path, so the
         dataset would have to live at `$HF_LEROBOT_HOME/<repo_id>` and be symlinked
         into place. `MixtureSource` carries `root`, so it stays where the converter
         wrote it.
      2. `run_yam.sh` stage [0] asserts `data.mixture` is non-empty and that the draws
         sum to batch_size. A single-source config fails the launcher outright.
    Sampling is unaffected: `StratifiedBatchSampler` with one source draws all 64
    indices from a random permutation of it, reshuffled on wrap -- ordinary shuffled
    training.

INHERITED AND CORRECT, so nothing is overridden here:
    The repack maps exactly the keys the converter emits
    (observation.images.{top,left_wrist,right_wrist} / observation.state / action), and
    the delta mask make_bool_mask(6, -1, 6, -1) matches the stored 14-D layout
    [L j0-5, L grip, R j0-5, R grip]. `use_delta_joint_actions` stays True: the
    converter writes raw teleop leader positions, i.e. ABSOLUTE actions.

NO HELD-OUT SPLIT
    `exclude_episodes=()` trains on 100% of the data, by request. There is therefore NO
    honest offline eval for this arm. For a number later, set `holdout_fraction` on the
    source (supported on this path) or score against a separately converted set.

NORM STATS MUST BE COMPUTED, not copied from yam7h_*. Different robot campaign,
different joint distribution -- rig A parks the left arm entirely, rigs B/C do not --
so the q01/q99 quantiles move.

ONE EPOCH, AND IT IS TIED TO batch_size. `len(LeRobotDataset)` is the frame count
(delta_timestamps clamps at episode ends rather than dropping samples), so
epochs = steps * batch_size / total_frames. The step count below is DERIVED from the
frame count in `configs/fd/datasets.py`, so changing batch_size recomputes it instead
of silently changing the epoch count. Verify the frame count after any reconversion:
    python -c "import json;i=json.load(open('/workspace/abc-ego-lerobot/meta/info.json'));print(i['total_frames'])"

batch_size 64 assumes 80GB-class hardware, as measured for the pi05_yam7h arms
(~10.5 GB/GPU of AdamW moments + grads for the 872.8M trainable set under
--fsdp-devices 1). H100 SXM 80GB clears this with room to spare.
"""

from configs._shared.arms import pi05_arm
from configs._shared.schedule import Schedule
from configs.fd import datasets as ds
from openpi.training import registry
import openpi.training.config as _config

_BATCH_SIZE = 64

# Exactly one epoch. warmup 500 is 4.4% of this run, against 2.1% of the 23.6k-step
# arms -- fine for a cosine schedule; drop to 250 if the first 500 steps look wasted.
SCHEDULE = Schedule.for_epochs(
    frames=ds.ABCEGO_SD_FRAMES,
    batch_size=_BATCH_SIZE,
    epochs=1.0,
    warmup_steps=500,
)

registry.register(
    pi05_arm(
        "pi05_abcego_sd",
        sources=(
            _config.MixtureSource(
                repo_id=ds.ABCEGO_SD_REPO,
                samples_per_batch=_BATCH_SIZE,  # == batch_size: the only source
                root=ds.ABCEGO_SD_ROOT,
                exclude_episodes=(),
            ),
        ),
        asset_id="abcego_sd",
        schedule=SCHEDULE,
        batch_size=_BATCH_SIZE,
        # BOTH augmentation stacks off. See `pi05_arm` -- this one flag kills the
        # data-side ImageAugmentConfig and openpi's built-in model-side stack together.
        augment=False,
        # Pin every 5,000th checkpoint permanently, so steps 5,000 and 10,000 survive
        # the rolling max_to_keep=4 window instead of being deleted by later saves.
        keep_period=5_000,
    )
)

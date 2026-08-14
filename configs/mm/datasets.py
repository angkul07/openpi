"""Datasets, roots and splits for mm (makermods), on SO-101.

Two pools, one task -- *"pick up the cube and place it in the tray"*:

    sim   ManiSkill SO-101, 50 episodes, 16,658 frames, 9 min 15 s, 30 Hz, 2 cameras
    ego   EgoDex clips retargeted to SO-101, 324 clips, ~20 min 15 s, 30 Hz

The sim numbers are MEASURED, from `meta/info.json` and `meta/episodes` of
`makermods/maniskill_50ep_so101_blue_cube_orange_tray_20260812_131142`. The ego numbers
are ESTIMATED and marked as such -- that dataset does not exist yet (see below), and
nothing load-bearing derives from them: the step count comes off the sim pool, which is
known exactly, so the estimate only moves the reported epoch table.

===========================================================================
NEITHER POOL IS READY TO TRAIN ON AS IT STANDS. Four things must happen first.
===========================================================================

1. SIM IS LeRobot v3.0; openpi READS v2.1 ONLY.
   `meta/info.json` says `"codebase_version": "v3.0"` and the layout is the v3 one
   (`data/chunk-000/file-000.parquet` holding many episodes, `meta/episodes/**.parquet`,
   `meta/tasks.parquet`). openpi pins lerobot at rev 0cf8648, whose `CODEBASE_VERSION`
   is `"v2.1"` and whose `check_version_compatibility` raises on anything else. Convert
   to v2.1 -- one parquet per episode under `data/chunk-000/episode_%06d.parquet`,
   `meta/episodes.jsonl`, `meta/tasks.jsonl` -- and point `SIM_ROOT` at the result.
   `SIM_HOLDOUT_EPISODES` assumes the conversion PRESERVES episode indices 0..49.

2. THE EGO POOL IS STILL RAW VIDEO.
   324 EgoDex clips: one bare 1920x1080 mpeg4 view, no camera key, and -- decisively --
   no action stream at all. dt-pipeline stage 6 has retargeted this corpus to SO-101
   (324/324 clips, 2.67 cm mean IK error), but the joint trajectories still need
   multiview synthesis and a v2.1 conversion before openpi can read them. Until then
   the two mixture arms cannot run; the sim-only arm can.

3. THE EGO CONVERSION HAS TO MATCH THE SIM SCHEMA EXACTLY, IN FOUR WAYS.
   The mixture applies ONE repack and ONE set of transforms to both sources, and the
   norm stats are computed over the blend. Anything that disagrees becomes contradictory
   supervision that no hyperparameter fixes (this is what the metres-vs-[0,1] gripper
   did to `pi05_ax91_mix9010`).

     a. COLUMN ORDER. `observation.state` and `action` must be
        `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`.
        Stage 6 emits a (T,12) two-arm frame with the left half parked, in the URDF's
        reverse-kinematic order (`wrist_roll ... shoulder_pan`) -- the exact reverse.
        Slice and remap; do not trust the raw order.
     b. GRIPPER UNITS. Sim's `gripper.pos` and the retargeted gripper must live on the
        same scale. Check before training, not after: quantile norm is computed over the
        mixture, so a pool whose gripper range sits entirely inside the other's collapses
        under normalisation and flow matching mode-averages the two into a half-open
        gripper.
     c. CAMERA KEYS. Exactly `observation.images.front` and `observation.images.wrist`,
        per `SO101` in `configs/_shared/robots.py`. EgoDex has one camera, so both are
        synthesised -- the Method-A' keypoint-guided crop used for the YAM multiview
        dataset applies directly (scene = square centre crop, wrist = crop tracking the
        projected grasp point).
     d. ASPECT RATIO -- the quiet one. `preprocess_observation` calls
        `resize_with_pad`, which PRESERVES aspect and letterboxes. Sim at 640x480 (4:3)
        becomes 224x168 content between two 28 px bars; ego at 1920x1080 (16:9) becomes
        224x126 between two 49 px bars. The halves would then differ in framing as well
        as in content. Make both square before openpi sees them -- centre-crop sim to
        480x480 and ego to 1080x1080, then store 224x224.

4. RE-ENCODE SIM OUT OF av1 WHILE CONVERTING.
   The sim videos are av1/yuv420p, which decodes far slower on CPU than h264 and is the
   likeliest cause if the loader turns out to be the bottleneck. Since openpi resizes
   everything to 224x224 anyway, storing 640x480 av1 is pure cost -- write 224x224 h264
   like the YAM pipeline does, which also settles (3d) for free.

Known defect, not blocking: the sim dataset has 731 more video frames than state rows
across 45 of its 50 episodes. LeRobot indexes by state rows, so training reads the
first N frames of each video and the surplus tail is never sampled -- harmless as long
as the extra frames are at the END. If they are at the start, every image is
misaligned with its state by a few frames and the whole thing is quietly wrong. Verify
alignment on one episode before spending GPU hours.
"""

from __future__ import annotations

import json
import os
import pathlib

_HERE = pathlib.Path(__file__).parent


# ---------------------------------------------------------------------------
# Sim -- ManiSkill SO-101
# ---------------------------------------------------------------------------

SIM_REPO = "makermods/maniskill_50ep_so101_blue_cube_orange_tray_20260812_131142"
# Point this at the v2.1 CONVERSION, not at a plain `hf download` of the repo above.
SIM_ROOT = os.environ.get("MM_SIM_ROOT", "/workspace/mm/sim_v21")

TASK = "pick up the cube and place it in the tray"  # the only task string in either pool


def _splits() -> dict:
    """The checked-in split manifest; regenerate with `configs/mm/make_sim_splits.py`."""
    return json.loads((_HERE / "sim_splits.json").read_text())


_SPLITS = _splits()

SIM_TOTAL_EPISODES = _SPLITS["total_episodes"]  # 50
SIM_TOTAL_FRAMES = _SPLITS["total_frames"]  # 16,658 -- 9.25 min at 30 Hz

# Episodes withheld from EVERY arm, so offline eval scores frames none of the three
# checkpoints ever saw and the comparison between them is honest. Picked by stride --
# see the rationale in make_sim_splits.py.
#
# It costs 10% of an already tiny pool, and 5 episodes is a thin eval set: treat gaps
# between arms of less than ~10% as a tie. It is still worth it. At the epoch counts
# these arms run (see `so101/sim_vs_ego.py`) the run WILL fit the training set, so
# "which checkpoint to ship" is a real question, and openpi has no in-training
# validation metric -- `holdout_fraction` and `exclude_episodes` only remove episodes
# from the training stream. Something has to be held back or checkpoint selection is
# blind. To spend the full pool on training instead, set this to `()`; every arm then
# picks it up, which is the point of it being one constant.
SIM_HOLDOUT_EPISODES: tuple[int, ...] = tuple(_SPLITS["holdout_episodes"])  # (4, 14, 24, 34, 44)
SIM_HOLDOUT_FRAMES = _SPLITS["holdout_frames"]  # 1,671 -- 10.03% of the pool

# What the full-sim arms train on: everything except the holdout.
SIM_TRAIN_FRAMES = _SPLITS["train_frames"]  # 14,987 -- 8.33 min

# What the half-size arm excludes: the holdout PLUS the episodes it drops, merged in the
# manifest so a config passes one list and cannot forget the holdout.
SIM_HALF_EXCLUDE_EPISODES: tuple[int, ...] = tuple(_SPLITS["half_exclude_episodes"])
SIM_HALF_TRAIN_FRAMES = _SPLITS["half_train_frames"]  # 7,756 -- 4.31 min, 51.75% of train


# ---------------------------------------------------------------------------
# Ego -- EgoDex retargeted to SO-101
# ---------------------------------------------------------------------------
# DOES NOT EXIST YET; see item 2 in the module docstring. The repo id is a label here --
# `LeRobotDatasetMetadata` resolves from `root` when one is given -- so what has to be
# right on the box is EGO_ROOT.
EGO_REPO = "makermods/egodex-so101-pickplace"
EGO_ROOT = os.environ.get("MM_EGO_ROOT", "/workspace/mm/ego_v21")

# ESTIMATED: 324 clips x 20 min 15 s at 30 Hz. Replace with the measured
# `total_frames` once the dataset is built. Nothing derives from it -- the schedule
# comes off the sim pool -- so it only affects the epoch table the module prints.
EGO_TOTAL_EPISODES = 324
EGO_TOTAL_FRAMES = 36_400  # ~20.2 min

# The mixture arms want 10 min and 5 min of ego out of that ~20 min. Both are taken with
# `MixtureSource.holdout_fraction`, which withholds a deterministic seeded sample of
# episodes, rather than by building two more datasets on disk.
#
# Two consequences to know about:
#   * The fraction is of EPISODES, not of frames. Across 324 clips the two agree closely
#     enough; across 20 they would not.
#   * The two subsets are INDEPENDENT draws, not nested -- `select_holdout_episodes`
#     calls `rng.choice(..., size=n)`, and choice at n=162 is not a prefix of choice at
#     n=243. The 5-minute pool is therefore not a subset of the 10-minute pool. Both are
#     uniform samples of the same corpus and the experimental variable is how MUCH ego
#     there is, so this is acceptable; if strict nesting is ever wanted, pre-build the
#     subsets and give them their own roots (the `yam7h`/`yam1090` pattern).
EGO_HALF_HOLDOUT_FRACTION = 0.50  # -> ~162 clips, ~10.1 min trained on
EGO_QUARTER_HOLDOUT_FRACTION = 0.75  # -> ~81 clips, ~5.1 min trained on
EGO_HOLDOUT_SEED = 0

EGO_HALF_TRAIN_FRAMES = round(EGO_TOTAL_FRAMES * (1 - EGO_HALF_HOLDOUT_FRACTION))  # ~18,200
EGO_QUARTER_TRAIN_FRAMES = round(EGO_TOTAL_FRAMES * (1 - EGO_QUARTER_HOLDOUT_FRACTION))  # ~9,100

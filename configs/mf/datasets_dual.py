"""Datasets, roots and splits for mf's DUAL-ARM Piper run.

NOT an extension of `configs/mf/datasets.py` -- that module describes the single-arm
7-D build under `/workspace/final/`, and nothing there is reusable here: different
schema (14-D, both arms live), different camera set, different pools.

Two pools under `/workspace/final_data/`, both LeRobot v2.1, both 20 Hz, both
`agilex_piper_bimanual`, both on ONE schema -- `float32[14]` =
`[right_j1-6, left_j1-6, right_gripper, left_gripper]` in DEGREES, grippers
normalized [0,1] (0 = closed), cameras `observation.images.top` +
`observation.images.left-arm` + `observation.images.right-arm`, video 480x640 h264:

    pool         eps    frames    duration    tasks   what it is
    -----------  ----   -------   ---------   -----   ---------------------------
    teleop_v21     86   138,120   1 h 55.1 m      1   REAL dual-arm rig
    ego_v21       377   141,302   1 h 57.8 m    241   human video, retargeted

The halves are budget-matched to 3,182 frames (~2.3%) of each other, so the 50/50
draw is near-true of the disk as well as of the gradient. Note the episode-length
asymmetry: teleop averages ~80 s/episode against ego's ~19 s -- 86 episodes is a
SMALL pool in clips even though it is not in frames, which is what sizes the holdout
below.

THE LAYOUT IS JOINTS-MAJOR (grippers at indices 12 and 13), which the default
arm-major `RobotSpec` mask would silently misparse -- `PIPER_DUAL` declares
`grippers_trailing=True` for exactly this build. See `configs/_shared/robots.py`.

WHAT IS ALREADY RECONCILED, per the final_data alignment table
---------------------------------------------------------------
Format, robot, rate, camera keys, video geometry (same 480x640 h264 on both halves --
the sideways-and-letterboxed failure of the single-arm mixture cannot recur; both
halves letterbox IDENTICALLY under `resize_with_pad`, so the framing channel
carries no half-identity), state/action dtype and layout, joint units (degrees),
gripper encoding.

TWO THINGS TO KNOW THAT NO SPEC FIXES
-------------------------------------
1. GRIPPER OCCUPANCY DIVERGED AND WAS RECONCILED AT THE DATA LAYER. The original
   alignment table flagged ego resting near-CLOSED (~0.25/0.26) against teleop
   resting OPEN (~0.73/0.81) -- a content divergence, not an encoding one (the
   channel is CONTINUOUS [0,1] on both halves, unlike the binarized single-arm
   build). It has since been fixed on the datasets by the client (see
   `vast_run/pi05/rescale_ego_gripper.py` for the tooling family). The numbers above
   are the PRE-fix table; the quantile preflight below is where a residual mismatch
   would show, so it doubles as the verification.

2. IMAGE SOURCE PARALLAX IS NOT SHARED, as on the single-arm mixture: teleop's three
   views are three PHYSICAL cameras; ego's three are synthesized crops of ONE
   egocentric frame and carry no parallax. No fix at the dataset layer -- it is what
   monocular retargeting gives. A wrist-view ablation reads as "the extra views are
   redundant on half the data", not as a bug.

ACTION CONVENTION differs between the halves the same benign way as single-arm: ego
is exact next-frame target (diff 0.0), teleop is the absolute command with ~1-frame
tracking lag. Both are absolute next-state actions, which is what
`use_delta_joint_actions=True` expects.

STILL WORTH THE 30-MINUTE PREFLIGHT before spending GPU-hours: per-dim p1/p50/p99 of
`observation.state` and `action`, teleop against ego, side by side over all 14 dims.
The alignment table says the schemas agree; it does not say the OCCUPANCY does, and
retarget wrist saturation compressing teleop's interior band under blend-computed
quantile norm is the `pi05_ax91_mix9010` failure mode, now available on two wrists.
"""

from __future__ import annotations

import os

# Root of the dual-arm build. One env var moves everything, as with `MF_FINAL_ROOT`.
FINAL_ROOT = os.environ.get("MF_DUAL_FINAL_ROOT", "/workspace/final_data")


def _pool(name: str) -> str:
    """Root for one pool, individually overridable (`MF_DUAL_TELEOP_V21_ROOT`, ...)."""
    return os.environ.get(f"MF_DUAL_{name.upper()}_ROOT", f"{FINAL_ROOT}/{name}")


# ---------------------------------------------------------------------------
# Teleop -- the real dual-arm rig
# ---------------------------------------------------------------------------
# `repo_id` is a LABEL (resolution is from `root`), but norm-stat lookup and the
# mixture sampler key on it, so it stays distinct per pool.
TELEOP_REPO = "mf/piper-dual-teleop-v21"
TELEOP_ROOT = _pool("teleop_v21")
TELEOP_EPISODES = 86
TELEOP_FRAMES = 138_120  # 1 h 55.1 m at 20 Hz; ~1,606 frames (~80 s) per episode

# The single teleop task string, and the prompt every eval should use.
# TBD: read the actual string from `<TELEOP_ROOT>/meta/tasks.jsonl` on the box -- the
# alignment table records the COUNT (1) but not the text.
TASK = "TBD"


# ---------------------------------------------------------------------------
# Retargeted -- human video, mapped onto the same dual-arm Piper
# ---------------------------------------------------------------------------
EGO_REPO = "mf/piper-dual-ego-v21"
EGO_ROOT = _pool("ego_v21")
EGO_EPISODES = 377
EGO_FRAMES = 141_302  # 1 h 57.8 m at 20 Hz; 241 distinct task strings


# ---------------------------------------------------------------------------
# Holdout -- teleop only, and a contiguous tail
# ---------------------------------------------------------------------------
# The single-arm doctrine carries over, because every reason does: openpi has no
# in-training validation, the run makes ~14-15 passes over each pool, and only a
# teleop holdout can distinguish "the late checkpoint is better" from "the late
# checkpoint has memorised more". A contiguous TAIL rather than a scattered draw,
# because consecutive episodes are near-duplicate takes from one session and a
# scattered split puts a near-twin of every held-out episode into training (the
# Piper H 72/79 failure). Full argument in `configs/mf/datasets.py`.
#
# EIGHT episodes, not the single-arm 15. That 15 was 5% of a 579-episode teleop pool;
# here it would be 17.4% of 86 -- withholding a sixth of the scarcer, headline-task
# half to score checkpoints is the tail wagging the dog. 8 is 9.3% of episodes
# (~12,850 frames, ~10.7 min), and since the pool is ONE task, eight ~80 s clips are
# near-interchangeable takes -- plenty to rank a checkpoint ladder, which is all this
# split is for.
#
# ASSUMES episodes are stored in recording order -- confirm on the box; if shuffled,
# pick indices by session from `meta/episodes.jsonl` instead.
#
# To spend the whole pool on training instead, set this to (). Nothing else changes.
_HOLDOUT_EPISODES = 8

TELEOP_HOLDOUT_EPISODES: tuple[int, ...] = tuple(
    range(TELEOP_EPISODES - _HOLDOUT_EPISODES, TELEOP_EPISODES)
)  # 78..85, 9.3% of the pool

# ESTIMATED at the pool's mean episode length; exact counts live in
# `meta/episodes.jsonl` on the box. Nothing load-bearing derives from this -- the step
# count is fixed at 30,000 -- it only moves the epoch table `describe()` prints.
TELEOP_TRAIN_FRAMES = round(TELEOP_FRAMES * (1 - _HOLDOUT_EPISODES / TELEOP_EPISODES))  # ~125,272

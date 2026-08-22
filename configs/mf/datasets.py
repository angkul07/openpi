"""Datasets, roots and splits for mf, on the single-arm Agilex Piper.

Four pools under `/workspace/final/`, all LeRobot v2.1, all 20 Hz, all `agilex_piper`,
all on ONE schema -- `float32[7]` = `[j1..j6 degrees, gripper in {0,1}]`, cameras
`observation.images.top` + `observation.images.right-arm`:

    pool                 eps          frames    duration   what it is
    -------------------  -----------  --------  ---------  ---------------------------
    mrfood1                294 (343)    61,709     51 min   REAL teleop, plate pick
    mrfood3                285 (332)    68,271     57 min   REAL teleop, plate pick
    piper_ego            1,574         109,730   1 h 31 m   EgoDex, retargeted to Piper
    piper_stera_plate      226          20,385     17 min   stera-10m, retargeted

The two halves are budget-matched ON PURPOSE and to a remarkable degree: teleop is
129,980 frames (1 h 48.3 m) against retargeted 130,115 (1 h 48.4 m), a gap of 135
frames. Teleop was trimmed to hit that -- mrfood1 is 294 of 343 episodes and mrfood3 285
of 332, whole episodes dropped in golden-ratio order so the cut spreads across recording
sessions instead of concentrating in one. That is what makes `mf_pi05_ea`'s 50/50 draw
mean the same thing at pool level and at gradient level; see `piper/ego_vs_teleop.py`.

The bracketed original episode counts are history, not an index space. Every pool here
is renumbered 0..N-1 by the v2.1 build -- see HOLDOUT below.

===========================================================================
ONE THING IS NOT READY, AND IT IS THE IMAGES. Fix it before launching.
===========================================================================

THE TWO HALVES DO NOT SHARE A VIDEO GEOMETRY.

    piper_ego, piper_stera_plate    224x224   h264   square, upright
    mrfood1, mrfood3                480x640   h264   portrait, rotated ~90 deg

Both differences bite, and neither raises:

  a. ASPECT. `preprocess_observation` calls `resize_with_pad`, which PRESERVES aspect
     and letterboxes. The retargeted half is already square, so it fills 224x224 edge to
     edge. mrfood at 480x640 becomes 168x224 of content between two 28 px bars. Half of
     every batch would then carry a framing the other half never shows, on a channel the
     model reads as content.
  b. ROTATION -- the worse one. mrfood's two views are rotated ~90 deg; the retargeted
     views are upright. So gravity points sideways in half the mixture. Image
     conditioning has no way to reconcile that, and no amount of augmentation is going
     to teach it that the two are the same scene class.

FIX AT CONVERSION TIME, NOT HERE: rotate mrfood upright, centre-crop to square, store
224x224 h264 -- which also settles (a) for free and matches what the retargeted half
already is. openpi resizes everything to 224 anyway, so nothing is lost.

This is deliberately NOT a `dataclasses.replace` or a transform. A preprocessing step
that only the training path applies would be absent at serving, and the policy would
then see, on the real rig, exactly the geometry it was never trained on.

THREE MORE THINGS TO CHECK, none of which error at load
-------------------------------------------------------
1. PARALLAX IS NOT SHARED. mrfood's `top` and `right-arm` are two PHYSICAL cameras. The
   retargeted pair are two crops of ONE egocentric frame (scene = square centre crop,
   wrist = a scale-free crop at 620/1080 of frame height), so they carry no parallax and
   no independent viewpoint at all. This one has no fix at the dataset layer -- it is
   what retargeting from monocular video gives you. Recorded so that a wrist-view
   ablation is read as "the second view is redundant on half the data" rather than as a
   bug.
2. THE `top` CAMERA ON THE mrfood RIG OFTEN CANNOT SEE THE PLATE -- the arm's black
   shroud fills it. That was measured on mrfood2 (78/347 episodes mislabelled by a VLM
   reading only `top`), which is why `right-arm` carries the task signal on the teleop
   half. Not a blocker; it is a reason not to be surprised if a `top`-only ablation
   collapses.
3. JOINT RANGE OCCUPANCY DIFFERS IN SHAPE. In shared degrees the retargeted joints reach
   every Piper limit while mrfood sits in a narrow interior band -- the known retarget
   saturation, driven by wrist joint5 (+/-1.22 rad against the +/-1.571 the retarget
   assumed). Quantile norm is computed over the BLEND, so mrfood's real working range
   gets compressed into the middle of a range set by the saturated half. Do the
   preflight in `piper/ego_vs_teleop.py` before spending the GPU-hours; this is the
   same class of defect that `pi05_ax91_mix9010` shipped.

THE GRIPPER IS ALREADY RECONCILED, and it took two steps
--------------------------------------------------------
Both halves are binary {0 = closed, 1 = open}. Getting there was not a threshold:
mrfood was already near-binary (1 transition/clip) and needed nothing, but the
retargeted gripper is a continuous human hand-aperture signal, and a plain threshold on
it gave a median of 3-4 transitions per clip -- phantom grasp/release events inside a
single pick-and-place. Per-clip min/max normalisation plus a Schmitt trigger
(0.40 +/-0.15, 8-frame dwell) gives median 2 (= grasp + release). Open-fractions land at
0.64/0.49 retargeted against 0.56/0.51 teleop.

Residual: stera is the weaker signal -- 10% of its clips still exceed 4 transitions.

The channel stays ABSOLUTE through the delta transform (see `PIPER_SINGLE`), so the
model predicts a level, not an edge. It predicts a CONTINUOUS level, though -- pi0.5 is
flow matching and has no notion of a discrete output -- so whatever drives the real
gripper has to threshold dim 6. Do not assume it comes back as exactly 0.0 or 1.0.

ACTION CONVENTION differs between the halves in a way that is fine but worth knowing:
the retargeted pools have `action[t] == state[t+1]` exactly (diff 0.0), while mrfood
bottoms out at lag 1 with ~0.04 deg residual, which is real tracking error rather than a
convention difference. Both are absolute next-state actions, which is what
`use_delta_joint_actions=True` expects.

TASK STRINGS are not balanced, and that is deliberate. mrfood is one string
(`pick_and_lift_right`); the retargeted half carries ~30 distinct strings on ego and
~127 on stera, read from each clip's hdf5 `description` attr. With
`prompt_from_task=True` half of every batch is therefore conditioned on a sentence the
eval never asks for. Leaving it that way makes the run answer "does DIVERSE retargeted
pick-place help the plate pick", which is the question; balancing it would answer a
different one.
"""

from __future__ import annotations

import os

# Root of the single-arm build. Every pool below hangs off it, so a box that lays the
# data out differently needs one env var rather than four.
FINAL_ROOT = os.environ.get("MF_FINAL_ROOT", "/workspace/final")


def _pool(name: str) -> str:
    """Root for one pool, individually overridable (`MF_MRFOOD1_ROOT`, ...)."""
    return os.environ.get(f"MF_{name.upper()}_ROOT", f"{FINAL_ROOT}/{name}")


# ---------------------------------------------------------------------------
# Teleop -- the real rig, the real task
# ---------------------------------------------------------------------------
# `repo_id` is a LABEL here, not something that gets downloaded:
# `LeRobotDatasetMetadata` resolves from `root` whenever one is given. What has to be
# right on the box is the root. The ids are still distinct per pool because norm-stat
# lookup and the mixture sampler key on them.
MRFOOD1_REPO = "mf/piper-mrfood1"
MRFOOD1_ROOT = _pool("mrfood1")
MRFOOD1_EPISODES = 294
MRFOOD1_FRAMES = 61_709  # 51.4 min at 20 Hz

MRFOOD3_REPO = "mf/piper-mrfood3"
MRFOOD3_ROOT = _pool("mrfood3")
MRFOOD3_EPISODES = 285
MRFOOD3_FRAMES = 68_271  # 56.9 min at 20 Hz

TELEOP_FRAMES = MRFOOD1_FRAMES + MRFOOD3_FRAMES  # 129,980 -- 1 h 48.3 m

# The task string the policy is actually being trained to do, and the prompt every eval
# should use. mrfood2 exists upstream and is deliberately NOT here: it was dropped from
# the single-arm build.
TASK = "pick_and_lift_right"


# ---------------------------------------------------------------------------
# Retargeted -- human video, mapped onto the same Piper
# ---------------------------------------------------------------------------
PIPER_EGO_REPO = "mf/piper-ego-retargeted"
PIPER_EGO_ROOT = _pool("piper_ego")
PIPER_EGO_EPISODES = 1_574
PIPER_EGO_FRAMES = 109_730  # 1 h 31.4 m at 20 Hz

# NOT 2 h 17 m. Resampling 30 -> 20 Hz preserves wall-clock, so dividing the PRE-resample
# 164,601 frames by the POST-resample rate overstates it by half. The duration is the
# one arithmetic here that has been got wrong before.

PIPER_STERA_REPO = "mf/piper-stera-plate-retargeted"
PIPER_STERA_ROOT = _pool("piper_stera_plate")
PIPER_STERA_EPISODES = 226
PIPER_STERA_FRAMES = 20_385  # 17.0 min at 20 Hz

EGO_FRAMES = PIPER_EGO_FRAMES + PIPER_STERA_FRAMES  # 130,115 -- 1 h 48.4 m


# ---------------------------------------------------------------------------
# Holdout -- teleop only, and a contiguous tail
# ---------------------------------------------------------------------------
# WHY HOLD ANYTHING OUT ON A SINGLE-ARM EXPERIMENT. `mf_pi05_ea` runs 70,000 steps,
# which is ~17-18 passes over each pool, and openpi has NO in-training validation metric
# -- `holdout_fraction` and `exclude_episodes` only remove episodes from the training
# stream. Without a holdout there is no number that distinguishes "step 70k is better"
# from "step 70k has memorised more", and the run keeps a checkpoint ladder precisely so
# that question can be asked. Training loss cannot answer it.
#
# WHY TELEOP ONLY. The headline metric is the plate pick on the real rig. The retargeted
# half is a training ingredient, never something scored, so nothing is withheld from it.
#
# WHY A CONTIGUOUS TAIL rather than `holdout_fraction`. `select_holdout_episodes` is a
# uniform `rng.choice`, and on this kind of capture consecutive episodes are near-
# duplicate takes from one session -- so a scattered split puts a near-twin of almost
# every held-out episode into training and flatters the model. That exact failure is
# documented on the Piper H teleop split, where the alternating cut left a holdout whose
# episodes were 72/79 training data for one of the two arms being compared. A tail block
# is the honest alternative when the split is not already fixed on disk.
#
# ASSUMES episodes are stored in recording order, so the tail is the last session(s)
# rather than a random 15. Confirm on the box before trusting the number; if the order
# turns out to be shuffled, the tail is no better than a random draw and the fix is to
# pick indices by session from `meta/episodes.jsonl` and list them here explicitly.
#
# INDEX SPACE: these are indices into the pool the arm actually reads, i.e. 0..293 for
# mrfood1 and 0..284 for mrfood3. The v2.1 build renumbered after filtering, so the
# original 343/332 numbering is NOT the index space -- an index from it either raises
# (out of range) or silently withholds an unrelated episode. `exclude_episodes` raises
# on out-of-range, which is what catches the first case loudly.
#
# To spend the whole pool on training instead, set both to (). Nothing else changes.
_HOLDOUT_PER_POOL = 15

MRFOOD1_HOLDOUT_EPISODES: tuple[int, ...] = tuple(
    range(MRFOOD1_EPISODES - _HOLDOUT_PER_POOL, MRFOOD1_EPISODES)
)  # 279..293, 5.1% of the pool
MRFOOD3_HOLDOUT_EPISODES: tuple[int, ...] = tuple(
    range(MRFOOD3_EPISODES - _HOLDOUT_PER_POOL, MRFOOD3_EPISODES)
)  # 270..284, 5.3% of the pool

# ESTIMATED, at each pool's mean episode length -- the exact per-episode frame counts
# are in `meta/episodes.jsonl` on the box and are not worth checking in. Nothing
# load-bearing derives from these: the step count is fixed at 70,000 by the experiment,
# so they only move the epoch table `describe()` prints.
MRFOOD1_TRAIN_FRAMES = round(MRFOOD1_FRAMES * (1 - _HOLDOUT_PER_POOL / MRFOOD1_EPISODES))  # ~58,561
MRFOOD3_TRAIN_FRAMES = round(MRFOOD3_FRAMES * (1 - _HOLDOUT_PER_POOL / MRFOOD3_EPISODES))  # ~64,678
TELEOP_TRAIN_FRAMES = MRFOOD1_TRAIN_FRAMES + MRFOOD3_TRAIN_FRAMES  # ~123,239 -- 1 h 42.7 m

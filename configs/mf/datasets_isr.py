"""Dataset, root and split for mf's 100%-TELEOP ISR-standardized handover run.

ONE pool this time -- no mixture, no budget-matching, no ego half. The dataset is
`Kavin60606/bimanual-handover-isr-std` on HF: the ISR-resampled, episode-curated
version of `PranayTest/bimanual-handover-2026-08-29` (real dual-arm Piper rig,
NOT the 08-26 capture behind `/workspace/final_data/teleop_v21` -- different
session, different episodes, do not mix the two up or their holdouts).

    pool              eps    frames    nominal dur   tasks   what it is
    --------------    ---    -------   -----------   -----   --------------------
    teleop_isr_v21     50     78,379      65.3 min       1   REAL dual-arm rig, ISR

Schema is the `final_data` dual-arm schema EXACTLY, so `PIPER_DUAL` applies
unchanged: `float32[14]` = `[right_j1-6, left_j1-6, right_gripper, left_gripper]`
-- joints-major, grippers TRAILING at indices 12/13 (`grippers_trailing=True` is
what parses it), joints in DEGREES, grippers continuous [0,1] (~1.0 open, ~0.1
closed-on-rim), cameras `observation.images.top` + `left-arm` + `right-arm`,
video 480x640 portrait, 20 Hz. Task string `right_pick_handover_left_place`,
verified from `meta/tasks.parquet` -- the same task as the dual mixture run.

WHAT WAS DONE TO THE DATA UPSTREAM (provenance: the HF README + build_results.json)
-----------------------------------------------------------------------------------
1. CURATION: 6 of 56 source episodes discarded (source indices 2, 9, 40, 45, 54,
   55 -- dead clips, single-arm-only clips, one cut before release). The 50
   survivors are RENUMBERED 0..49 in recording order; all indices below are in
   the renumbered space.
2. ISR RESAMPLING (Yang et al., arXiv:2606.22907, paper-default knobs): frames
   kept at equal gaps of accumulated motion+acceleration information rather than
   equal gaps of time, gripper open/close windows force-kept, then RESTAMPED onto
   a uniform 20 fps grid. 78,379 of 102,266 clean-episode frames kept (76.6%;
   per-episode 50-92%, idle-heavy episodes compress hardest).

TWO CONSEQUENCES OF ISR THAT NO SPEC ENCODES
--------------------------------------------
1. THE CLOCK IS INFORMATION, NOT WALL TIME. "20 Hz" is a restamped label: pauses
   are collapsed to nothing and fast motion is densely sampled, so a 50-step
   chunk spans MORE real motion than the 2.5 s the rate implies, and per-step
   deltas are larger and more uniform than raw teleop's. That is the point of
   ISR ([[isr_resampling_eval]]: apply to teleop, skip on ego). It also means
   "65.3 min" above is nominal (frames/rate); the kept source wall-clock is
   ~85.2 min.
2. THE SOURCE'S ~2-FRAME CONTROL DELAY IS NOW A VARIABLE LAG. The HF README
   suggests aligning `action[t]` with `state[t+2]` -- that prescription is only
   coherent on the SOURCE's uniform grid. ISR drops frames non-uniformly, so on
   this dataset the state-lags-action offset varies frame to frame and a fixed
   shift would miscorrect most of it. Accepted as-is: the same class of benign
   tracking lag the `final_data` build documents at ~1 frame.

NOT READY TO TRAIN AS PUBLISHED -- TWO CONVERSION STEPS FIRST
-------------------------------------------------------------
The HF repo is LeRobot v3.0 (aggregated `file-000.parquet`, AV1 video). openpi's
loader consumes v2.1, and every prior conversion in this project went through the
same gap (the forge eval's v2.1-writer finding; the mrfood v3.0->v2.1 ports). So
before this config can run:
  a. convert v3.0 -> v2.1 under `ROOT` below (episode indices already 0..49);
  b. re-encode AV1 -> h264 while at it. Every measured num_workers/decode number
     in this project is h264; AV1 decode through pyav is slower and untested here.

ONE THING TO EYEBALL, NOT ASSUME: this rig family has a history of the `top`
camera being recorded 90-degrees sideways ([[mrfood_standardised_teleop]], on the
08-26 sibling). A uniformly-rotated view is self-consistent for a single-pool run
-- train and serving just have to agree -- but extract a frame and LOOK before
wiring the eval, and note the portrait 480x640 letterboxes to 168x224 of content
under `resize_with_pad`; the eval must feed the same geometry.

WHAT IS SIMPLER THAN THE MIXTURE RUNS, and worth saying out loud: quantile norm
is computed over THIS pool alone. There is no retargeted half whose wrist
saturation can compress teleop's working band (the `pi05_ax91_mix9010` failure
mode), so the 14-dim preflight is a sanity check here, not a launch gate.
"""

from __future__ import annotations

import os

# v2.1 conversion target on the box. One env var moves it, as with the other builds.
ROOT = os.environ.get("MF_ISR_TELEOP_ROOT", "/workspace/isr_std/teleop_isr_v21")

# `repo_id` is a LABEL (resolution is from `root`), but norm-stat lookup keys on it,
# so it stays distinct from `mf/piper-dual-teleop-v21` -- different capture, different
# distribution, never shared stats.
REPO = "mf/piper-dual-teleop-isr"

EPISODES = 50
FRAMES = 78_379  # 65.3 min nominal at the restamped 20 Hz; ~1,568 frames/episode mean

# The single task string, and the prompt every eval must use. From meta/tasks.parquet.
TASK = "right_pick_handover_left_place"


# ---------------------------------------------------------------------------
# Holdout -- a contiguous tail, as always
# ---------------------------------------------------------------------------
# This run is 100% teleop, so the holdout is not optional bookkeeping: openpi has no
# in-training validation, and scoring saved checkpoints against held-out episodes of
# THIS pool is the only way to rank the ladder. A contiguous TAIL rather than a
# scattered draw, for the standing reason: consecutive episodes are near-duplicate
# takes from one session, and a scattered split puts a near-twin of every held-out
# episode into training (the Piper H 72/79 failure).
#
# FIVE episodes = 10% of 50 -- the dual run's 8/86 (9.3%) ratio carried over. The pool
# is ONE task, so five ~80 s takes are near-interchangeable and plenty to rank a
# checkpoint ladder, which is all this split is for.
#
# ASSUMES the renumbered 0..49 preserves recording order (the curation drops episodes,
# it does not shuffle them) -- confirm against `meta/episodes` on the box; if shuffled,
# pick indices by session instead.
#
# To spend the whole pool on training instead, set this to (). Nothing else changes.
_HOLDOUT_EPISODES = 5

HOLDOUT_EPISODES: tuple[int, ...] = tuple(range(EPISODES - _HOLDOUT_EPISODES, EPISODES))  # 45..49

# ESTIMATED at the pool's mean episode length; exact counts live in `meta/episodes` on
# the box. NOTE this estimate is mildly load-bearing here, unlike the dual run: the arm
# derives its step count from it via `Schedule.for_epochs`. ISR's 50-92% per-episode
# keep ratios mean the true tail could be a few percent off the mean -- that moves the
# step count by the same few percent, which is noise against a 15-epoch target.
TRAIN_FRAMES = round(FRAMES * (1 - _HOLDOUT_EPISODES / EPISODES))  # ~70,541

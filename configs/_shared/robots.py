"""The embodiments we train on.

A `RobotSpec` states an embodiment once -- cameras, joint layout, dataset column names
-- and `LeRobotRobotDataConfig` derives the repack map, the delta mask and the policy
transforms from it. See `openpi.policies.robot_policy` for the two non-obvious rules
these specs have to respect (slot ORDER carries signal; a padding slot is masked for
PI0/PI05 but not PI0_FAST).

These live in the client tree rather than in openpi because they describe OUR robots.
A spec is reusable across clients, though -- two clients on the same rig share one. A
client with a bespoke arm can define its own in `configs/<client>/robots.py`.
"""

from __future__ import annotations

from openpi.policies.robot_policy import RobotSpec

# ---------------------------------------------------------------------------
# YAM -- 14-DoF bimanual, 3 real cameras
# ---------------------------------------------------------------------------
# All three cameras are real, so every image_mask is True and no slot is padding.
# State/action layout is [L j0-5, L grip, R j0-5, R grip].
#
# NOTE the 50/50 merged dataset (`pi05_50run_ea`) is ordered right-arm-first,
# [R j1-6, R grip, L j1-6, L grip]. That is still correct against this spec: the delta
# mask is symmetric across the two arms, so only a layout with different PER-ARM
# structure would need its own spec.
YAM = RobotSpec(
    name="yam",
    cameras=(
        "observation.images.top",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    ),
    arms=2,
    joints_per_arm=6,
    gripper_per_arm=True,
    # 30 Hz. Verified against every YAM dataset we train on, from frames/hours in
    # meta/info.json: abc-teleop 7h subset 251,703 / 2.331 h, ego multiview
    # 1,074,893 / 9.953 h, abc-ego screwdriver 730,496 / 6.76 h, 50/50 merge
    # 755,964 / 7.00 h -- all 30.0.
    # NOT verified for the legacy `Kavin60606/yam_pi0fast_train` set behind the two
    # pi0_fast_yam arms; if it turns out to differ the loader will say so on the box,
    # and the fix is a per-config dataclasses.replace(YAM, control_hz=...).
    # A 50-step chunk is 1.67 s of future here.
    control_hz=30.0,
)


# ---------------------------------------------------------------------------
# Piper H -- 14-DoF bimanual, TWO real cameras of three stored
# ---------------------------------------------------------------------------
# THE STORED CAMERA KEY NAMES DO NOT DESCRIBE WHAT THE CAMERAS SEE. Established by
# extracting frames and looking, not by reading metadata (rig capture
# `PranayTest/pick-right-2026-08-06-w2-trial-01`):
#
#     key      teleop content              ego content                used as
#     -------  --------------------------  -------------------------  ----------------
#     front    top-down workspace view     full egocentric frame      base_0_rgb
#     right    right-arm wrist camera      tight crop on grasp point  left_wrist_0_rgb
#     top      SIDEWAYS view of the robot  wide crop on grasp point   DROPPED
#
# `front` and `right` genuinely correspond across the two halves of the mixture.
# `top` cannot: a head-mounted ego camera can never produce a sideways view of the
# robot, so that slot would hold two unrelated images under one key and the image
# conditioning would fight itself. Omitting it here is what drops it -- the repack
# discards any feature the spec does not name.
#
# The wrist view goes in SLOT 1, with slot 2 as the padding slot, because openpi's own
# two-camera policies (droid, libero) fill the leading slots and mask the trailing one.
# That the physical camera is on the RIGHT arm is not a reason to put it in
# `right_wrist_0_rgb`: the key is an identifier, not something the model knows the
# semantics of. This capture is also effectively single-arm -- the left arm never moves
# -- which is exactly the DROID configuration slots 0/1 encode.
#
# Piper H is 6-DoF + gripper per arm, so the 14-D layout matches YAM's. Both halves are
# 20 Hz, against 30 Hz everywhere in the YAM work.
PIPER_H = RobotSpec(
    name="piper_h",
    cameras=(
        "observation.images.front",
        "observation.images.right",
        None,  # `observation.images.top` deliberately dropped -- see above.
    ),
    arms=2,
    joints_per_arm=6,
    gripper_per_arm=True,
    # 20 Hz on BOTH halves -- ego 47,953 frames / 39.96 min and teleop 25,075 / 20.90
    # min both work out to 20.0. This is the one place the rate differs from the YAM
    # work, and it is why the constant is declared rather than assumed: a 50-step
    # chunk is 2.5 s of future here against YAM's 1.67 s, so the two are NOT
    # comparable at equal action_horizon even though the number looks the same.
    control_hz=20.0,
)


# ---------------------------------------------------------------------------
# SO-101 -- 6-DoF SINGLE arm, 2 real cameras
# ---------------------------------------------------------------------------
# The first single-arm spec here, and the first with fewer than 6 joints. Both fall out
# of `arms`/`joints_per_arm` with no special-casing:
#
#     action_dim = 1 * (5 + 1)           = 6
#     delta mask = make_bool_mask(5, -1) = (T,T,T,T,T,F)
#
# 5 arm DOF, NOT 6. The columns are `shoulder_pan, shoulder_lift, elbow_flex,
# wrist_flex, wrist_roll, gripper` -- there is no shoulder_roll. Worth stating because
# the dt-pipeline's retargeting stages M3-M6 still assume 6 joints + gripper and emit a
# (T,12) two-arm frame with the left half parked, so whatever converts that to LeRobot
# has to slice the 6 real columns out and order them as above. The SO-101 URDF's own
# joint order is reverse-kinematic (`wrist_roll ... shoulder_pan`), i.e. exactly
# backwards from the column order, so a retargeted source that skips the remap is not
# merely miscalibrated -- it is joint-reversed, and normalisation cannot recover it.
#
# Slot 1 holds the wrist camera and slot 2 is padding, per the leading-slots rule in
# `openpi.policies.robot_policy`. This is the DROID/libero occupancy pattern.
#
# 30 Hz, read from `meta/info.json` of
# `makermods/maniskill_50ep_so101_blue_cube_orange_tray_20260812_131142`
# (16,658 frames / 50 episodes / fps 30). A 50-step chunk is 1.67 s of future, the same
# as YAM -- so action_horizon IS comparable across those two, unlike Piper H.
SO101 = RobotSpec(
    name="so101",
    cameras=(
        "observation.images.front",  # -> base_0_rgb
        "observation.images.wrist",  # -> left_wrist_0_rgb
        None,  # padding slot
    ),
    arms=1,
    joints_per_arm=5,
    gripper_per_arm=True,
    control_hz=30.0,
)


# ---------------------------------------------------------------------------
# Piper (single arm) -- 6-DoF + gripper, 2 real cameras
# ---------------------------------------------------------------------------
# NOT `PIPER_H` above. That spec is the BIMANUAL 14-D rig with the `front`/`right`/`top`
# camera set. This one is the single-arm `agilex_piper` build under `/workspace/final/`,
# where the dead left half was dropped at conversion time rather than parked:
#
#     float32[7] = [j1..j6 DEGREES, gripper in {0, 1}]
#
# Dropping the parked half is what moves the gripper to index 6 in every pool at once.
# Before that, retargeted data carried it at 6 and mrfood teleop at 12 under one
# identical `float32[14]` dtype -- a mismatch nothing in the loader can see. Keep the
# two specs separate: a 14-D dataset read through this spec silently trains on the
# first 7 columns.
#
#     action_dim = 1 * (6 + 1)           = 7
#     delta mask = make_bool_mask(6, -1) = (T,T,T,T,T,T,F)
#
# The gripper stays ABSOLUTE, which is what makes the binarisation usable: a {0,1}
# channel differenced against the previous state would be {-1,0,+1} and the two edges
# would be two rare classes instead of one level the model holds.
#
# JOINTS ARE IN DEGREES, not radians. Nothing here converts them and nothing needs to --
# quantile norm is scale-free and every pool agrees. It matters when comparing to any
# YAM or Piper H checkpoint, whose action space is radians: the numbers are 57.3x apart
# and an eval that mixes the two reports nonsense.
#
# CAMERAS. `top` is the workspace view and `right-arm` is the wrist view, and unlike the
# Piper H rig these names were checked against content. Note the HYPHEN in `right-arm`;
# mrfood3 originally spelled it `r-arm` and was normalised on conversion, so a pool that
# missed that normalisation fails loudly in the repack rather than training blind.
#
# The wrist view goes in SLOT 1 with slot 2 as padding, per the leading-slots rule at
# the top of this file -- the DROID/libero two-camera occupancy pattern.
#
# WHAT THE TWO VIEWS ARE IS NOT THE SAME ON BOTH HALVES OF THE MIXTURE, and no spec can
# fix it: mrfood's pair are two PHYSICAL cameras with real parallax, while the
# retargeted pair are two crops of ONE egocentric frame and carry none. See
# `configs/mf/datasets.py`.
PIPER_SINGLE = RobotSpec(
    name="piper_single",
    cameras=(
        "observation.images.top",  # -> base_0_rgb
        "observation.images.right-arm",  # -> left_wrist_0_rgb
        None,  # padding slot
    ),
    arms=1,
    joints_per_arm=6,
    gripper_per_arm=True,
    # 20 Hz on all four pools, and for two different reasons. mrfood was RECORDED at 20;
    # the retargeted pools were resampled 30 -> 20 onto a uniform 50 ms grid (nearest-
    # frame decimation would have left alternating 33/67 ms gaps and a sawtooth velocity).
    # Cross-checked against the frame counts: ego 109,730 / 1 h 31 m, stera 20,385 / 17 m,
    # mrfood1 61,709 / 51 m, mrfood3 68,271 / 57 m all land on 20.0.
    #
    # A 50-step chunk is 2.5 s of future here -- same as PIPER_H, and NOT comparable to
    # the 1.67 s that the same `action_horizon` buys on YAM or SO-101.
    control_hz=20.0,
)


# ---------------------------------------------------------------------------
# Piper (dual arm) -- 14-DoF bimanual, 3 real cameras
# ---------------------------------------------------------------------------
# The THIRD Piper spec, and not a merge of the other two. `PIPER_H` is the old bimanual
# rig whose camera keys lie about content (`front`/`right`/`top`, one dropped);
# `PIPER_SINGLE` is the 7-D build with the dead arm removed at conversion. This is the
# new dual-arm build where both arms are live and the three stored views are the three
# the spec names -- YAM's occupancy pattern (all slots real, nothing padded, nothing
# masked), on Piper's 20 Hz.
#
# Note the HYPHENS: `left-arm` / `right-arm`, matching the single-arm `right-arm`
# convention (mrfood3's `r-arm` was normalised to it at conversion). A pool that stores
# underscores instead fails loudly in the repack, which is the good outcome.
#
# Slot assignment is semantic here and genuinely corresponds: `top` is the workspace
# view -> base_0_rgb, and each wrist camera goes to its own side's slot. That is more
# than the identifiers require (see PIPER_H, where a right-arm camera correctly sits in
# `left_wrist_0_rgb`) but there is no reason to be clever when the rig actually has the
# geometry the keys describe.
#
# THE LAYOUT IS JOINTS-MAJOR, NOT ARM-MAJOR, and it is why `grippers_trailing` exists.
# The `/workspace/final_data/` build stores
#
#     [right_j1-6, left_j1-6, right_gripper, left_gripper]
#
# on BOTH halves -- grippers at indices 12 and 13, not at 6 and 13. Read through the
# default arm-major assumption, the delta mask would difference `right_gripper` as if
# it were a joint and hold `left_j1` absolute, and nothing would raise. With the flag:
#
#     action_dim = 2 * (6 + 1)              = 14
#     delta mask = make_bool_mask(12, -2)   = (12xT, F, F)
#
# The layout is also RIGHT-ARM-FIRST, which is fine -- the mask is symmetric across
# arms either way. What is NOT enforceable from here is that every pool in a mixture
# agrees on that order: an L-first pool and an R-first pool both pass every check
# while disagreeing about which physical arm dims 0-5 drive. The final_data build
# states one order for both halves; if a new pool joins, verify against content
# (move one arm, watch which columns move), not metadata.
#
# 20 Hz as on every Piper build; a 50-step chunk is 2.5 s of future, NOT comparable to
# the 1.67 s the same `action_horizon` buys on YAM even though both are 14-D bimanual.
PIPER_DUAL = RobotSpec(
    name="piper_dual",
    cameras=(
        "observation.images.top",  # -> base_0_rgb
        "observation.images.left-arm",  # -> left_wrist_0_rgb
        "observation.images.right-arm",  # -> right_wrist_0_rgb
    ),
    arms=2,
    joints_per_arm=6,
    gripper_per_arm=True,
    grippers_trailing=True,  # [R j1-6, L j1-6, R grip, L grip] -- see above.
    control_hz=20.0,
)

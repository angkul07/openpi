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
)

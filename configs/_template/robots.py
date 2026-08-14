"""Embodiment spec for <CLIENT>'s robot.

Only needed if this client's rig is not already described in
`configs/_shared/robots.py`. A spec is reusable across clients -- two clients on the
same rig should share one -- so check there first and only add here if the rig is
genuinely bespoke.

Read `openpi.policies.robot_policy` before filling this in. Two rules are load-bearing:

  SLOT ORDER CARRIES SIGNAL. `cameras` is ordered to match the model's image slots
  (base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb). Real cameras fill the LEADING
  slots and padding trails -- openpi's own two-camera policies mask only the trailing
  slot, so a gap in the middle is an occupancy pattern the pretrained model has never
  seen. `RobotSpec` rejects it.

  DROPPING A CAMERA MEANS OMITTING IT. A feature not named here never enters the
  repack. Masking is not the same thing: a masked slot still pays full SigLIP cost,
  because the model embeds every image and applies masks only to the attention mask.
  (Neither skips the video DECODE -- LeRobotDataset decodes every video feature before
  the repack runs. Only not writing the key at conversion time avoids that.)

CHECK WHAT THE CAMERAS ACTUALLY SEE. On the Piper H rig the stored key names did not
describe the content -- `front` was a top-down view, `top` was sideways -- and the
mismatch was only found by extracting frames and looking. Do that before trusting
metadata, especially when two data sources are being mixed under one schema: a slot
whose content does not correspond across sources makes the image conditioning fight
itself.
"""

from __future__ import annotations

from openpi.policies.robot_policy import RobotSpec

CLIENT_ROBOT = RobotSpec(
    name="<client>_<rig>",
    cameras=(
        "observation.images.<scene>",  # -> base_0_rgb
        "observation.images.<wrist>",  # -> left_wrist_0_rgb
        None,  # padding slot; use a third real camera if the rig has one
    ),
    # Joint layout. action_dim and the delta mask are derived from these:
    #   action_dim = arms * (joints_per_arm + gripper)
    #   delta mask = arm joints -> delta, grippers -> absolute
    # Assumes an arm-major layout: [arm0 joints, arm0 gripper, arm1 joints, arm1 gripper].
    arms=2,
    joints_per_arm=6,
    gripper_per_arm=True,
    # LeRobot feature names, if this client's converter used different ones.
    state_feature="observation.state",
    action_feature="action",
)

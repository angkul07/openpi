"""Piper H policy transforms for openpi.

Adapted from `yam_policy.py` for the bimanual Piper H rig behind the 1-hour
ego+teleop mixture: 613 retargeted EgoDex clips (40 min) + 154 real teleop
episodes (21 min), both LeRobot v2.1, both 20 Hz, both 14-D state/action
([L j0-5, L grip, R j0-5, R grip]).

TWO REAL VIEWS, NOT THREE -- and the camera key names do not describe what the
cameras see. Established by extracting frames and looking, not by reading
metadata (rig capture `PranayTest/pick-right-2026-08-06-w2-trial-01`):

    key      teleop content                ego content                  used as
    -------  ----------------------------  ---------------------------  -----------------
    front    top-down workspace view       full egocentric frame        base_0_rgb
    right    right-arm wrist camera        tight crop on grasp point    left_wrist_0_rgb
    top      SIDEWAYS view of the robot    wide crop on grasp point     DROPPED

`front` and `right` genuinely correspond across the two halves. `top` cannot: a
head-mounted ego camera can never produce a sideways view of the robot, so that
slot holds two unrelated images under one key. Feeding it would make the two
halves of the mixture disagree about what the slot means, and the image
conditioning fights itself. It is dropped in the repack (see
`LeRobotPiperMixtureDataConfig`), not masked here -- masking still pays the full
SigLIP cost, because `Pi0.embed_prefix` embeds every entry of `obs.images` and
applies `image_masks` only to the attention mask.

WHY THE WRIST VIEW GOES IN SLOT 1 AND SLOT 2 IS THE PADDING SLOT.
`_model.IMAGE_KEYS` is an ordered tuple and `embed_prefix` concatenates in that
order, so slot position carries signal. Every openpi reference policy with fewer
than three real cameras fills slots 0 and 1 and masks the TRAILING slot --
`droid_policy` does `(True, True, False)`, `libero_policy` does the same with an
explicit `np.zeros_like(base_image)` in slot 2. pi0.5 pretraining has therefore
seen "base + one wrist real, right_wrist masked" far more often than the
alternative. Putting the wrist view in slot 2 and masking slot 1 would be a slot
occupancy pattern learned from scratch on one hour of data.

That the physical camera is the RIGHT arm's wrist is not a reason to put it in
`right_wrist_0_rgb`: the key is an identifier, not something the model knows the
semantics of. This capture is also effectively single-arm -- the left arm never
moves -- which is exactly the DROID configuration that slots 0/1 encode.
"""

import dataclasses

import einops
import numpy as np

from openpi.models import model as _model
import openpi.transforms as _transforms


def make_piper_example() -> dict:
    """Random input example for the Piper policy (used for dummy inference)."""
    return {
        "observation/front_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.rand(14),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:  # (c, h, w) -> (h, w, c)
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class PiperInputs(_transforms.DataTransformFn):
    # PI0/PI05 mask the padding image; PI0_FAST does not (see droid_policy).
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        scene = _parse_image(data["observation/front_image"])  # top-down / egocentric
        wrist = _parse_image(data["observation/right_image"])  # right wrist / grasp crop

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": scene,
                "left_wrist_0_rgb": wrist,
                # Padding slot. The model always expects three images
                # (`_model.IMAGE_KEYS`), so send zeros and mask them off.
                "right_wrist_0_rgb": np.zeros_like(scene),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class PiperOutputs(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Full 14-dim bimanual action; the model pads to action_dim=32 internally.
        return {"actions": np.asarray(data["actions"][..., :14])}

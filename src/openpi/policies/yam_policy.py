"""
YAM policy transforms for openpi.

WHERE THIS GOES:
    Copy this file into YOUR openpi fork at:  src/openpi/policies/yam_policy.py

It is adapted from src/openpi/policies/libero_policy.py for the YAM bimanual robot:
  - 3 cameras: top, left_wrist, right_wrist  (each 224x224x3)
  - 14-dim state, 14-dim action  (2 arms x 7)

Notes:
  - Unlike LIBERO (2 cams, one masked), all THREE YAM cameras are real, so every
    image_mask is True.
  - No manual state padding here -- openpi's model_transforms pad to the model dim.
    We just pass the 14-dim state/actions straight through.
"""

import dataclasses

import einops
import numpy as np

from openpi.models import model as _model
import openpi.transforms as _transforms


def make_yam_example() -> dict:
    """Random input example for the YAM policy (used for dummy inference)."""
    return {
        "observation/top_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/left_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
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
class YamInputs(_transforms.DataTransformFn):
    # PI0 vs PI0_FAST changes how masks/tokens are handled downstream.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": _parse_image(data["observation/top_image"]),
                "left_wrist_0_rgb": _parse_image(data["observation/left_wrist_image"]),
                "right_wrist_0_rgb": _parse_image(data["observation/right_wrist_image"]),
            },
            # All three YAM cameras are real -> all masks True.
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class YamOutputs(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Return the full 14-dim bimanual action (LIBERO used :7).
        return {"actions": np.asarray(data["actions"][..., :14])}

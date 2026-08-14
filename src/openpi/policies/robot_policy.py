"""Embodiment-parameterised policy transforms.

This replaces the per-robot `<name>_policy.py` + `LeRobot<Name>DataConfig` pair. Those
two files were ~95% identical between YAM and Piper H; the only real differences were
which dataset camera key feeds which model image slot, and how many slots are real.
Everything else -- the 14-D `[6 joints, gripper] x 2` layout, the delta mask, the
`action` column name, the `[..., :14]` output slice -- was copied verbatim, which is
exactly the kind of duplication that drifts.

A `RobotSpec` states the embodiment once. `RobotInputs`/`RobotOutputs` and the repack
map are derived from it.

TWO THINGS HERE ARE LOAD-BEARING AND NON-OBVIOUS.

1. SLOT ORDER CARRIES SIGNAL. `_model.IMAGE_KEYS` is an ordered tuple and
   `Pi0.embed_prefix` concatenates in that order, so which slot an image lands in is
   part of what the model learned in pretraining. Every openpi reference policy with
   fewer than three real cameras fills slots 0 and 1 and masks the TRAILING slot --
   `droid_policy` and `libero_policy` both do `(True, True, False)`. A two-camera robot
   should therefore declare `("scene", "wrist", None)`, NOT `("scene", None, "wrist")`:
   the latter is a slot-occupancy pattern the model has never seen. `RobotSpec` refuses
   the latter for that reason.

   That a physical camera is mounted on the RIGHT arm is not a reason to put it in
   `right_wrist_0_rgb`. The key is an identifier; the model does not know its semantics.

2. A PADDING SLOT IS MASKED FOR PI0/PI05 BUT NOT FOR PI0_FAST. That asymmetry is
   openpi's own convention (see `droid_policy`: "We don't mask out padding images for
   FAST models"), and it is reproduced here rather than reasoned about per robot.

   Masking does NOT save compute. `embed_prefix` embeds every entry of `obs.images` and
   applies `image_masks` only to the attention mask, so a masked slot still pays full
   SigLIP cost. To actually drop a camera, leave it out of the spec -- then it never
   enters the repack. (Even that does not skip its video DECODE: `LeRobotDataset`
   decodes every video feature before the repack runs. Not writing the key at
   conversion time is the only way to avoid that cost.)
"""

from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi.models import model as _model
import openpi.transforms as _transforms

# Intermediate keys produced by the repack and consumed by `RobotInputs`. They are
# derived mechanically from the model slot name so that no robot-specific spelling
# (`observation/top_image`, `observation/front_image`, ...) leaks into the pipeline.
_IMAGE_PREFIX = "observation/image/"
STATE_KEY = "observation/state"
ACTIONS_KEY = "actions"
PROMPT_KEY = "prompt"


def image_key(slot: str) -> str:
    """Intermediate repack key for a model image slot."""
    return f"{_IMAGE_PREFIX}{slot}"


@dataclasses.dataclass(frozen=True)
class RobotSpec:
    """One embodiment: its cameras, its joint layout, its dataset column names."""

    name: str

    # One entry per model image slot, in `_model.IMAGE_KEYS` ORDER. The value is the
    # LeRobot feature name that feeds that slot, or None for a padding slot.
    # Real cameras must come first -- see the module docstring.
    cameras: tuple[str | None, ...]

    # Joint layout. `action_dim` and the delta mask are derived from these.
    arms: int = 2
    joints_per_arm: int = 6
    gripper_per_arm: bool = True

    # LeRobot feature names.
    state_feature: str = "observation.state"
    action_feature: str = "action"

    def __post_init__(self) -> None:
        if len(self.cameras) != len(_model.IMAGE_KEYS):
            raise ValueError(
                f"{self.name}: cameras must have exactly {len(_model.IMAGE_KEYS)} entries, one per model "
                f"slot {_model.IMAGE_KEYS}, using None for a padding slot. Got {len(self.cameras)}."
            )
        real = [c is not None for c in self.cameras]
        if not any(real):
            raise ValueError(f"{self.name}: at least one camera slot must be a real camera.")
        # Real slots must be a prefix: (real, real, None) is fine, (real, None, real) is not.
        if real != sorted(real, reverse=True):
            pairs = zip(_model.IMAGE_KEYS, self.cameras, strict=True)
            filled = ", ".join(f"{k}={'real' if c else 'PAD'}" for k, c in pairs)
            raise ValueError(
                f"{self.name}: real cameras must fill the leading slots and padding must trail ({filled}). "
                "Slot order carries signal -- openpi's own policies mask only the trailing slot, so a "
                "gap in the middle is an occupancy pattern the pretrained model has never seen."
            )
        if self.arms < 1 or self.joints_per_arm < 1:
            raise ValueError(f"{self.name}: arms and joints_per_arm must be >= 1.")

    @property
    def dofs_per_arm(self) -> int:
        return self.joints_per_arm + (1 if self.gripper_per_arm else 0)

    @property
    def action_dim(self) -> int:
        """Real action width. NOT the model's `action_dim`, which is padded (32 on pi0.5)."""
        return self.arms * self.dofs_per_arm

    @property
    def real_cameras(self) -> tuple[tuple[str, str], ...]:
        """(model slot, dataset feature) for every slot that has a real camera."""
        pairs = zip(_model.IMAGE_KEYS, self.cameras, strict=True)
        return tuple((slot, feat) for slot, feat in pairs if feat is not None)

    def delta_action_mask(self) -> tuple[bool, ...]:
        """True where an action dim is converted to a delta relative to current state.

        Arm joints become deltas; grippers stay absolute. For the usual bimanual
        6-DoF-plus-gripper robot this is `make_bool_mask(6, -1, 6, -1)`.

        ASSUMES the state/action layout is arm-major:
            [arm0 joints..., arm0 gripper, arm1 joints..., arm1 gripper]
        A dataset ordered differently needs a different mask. Note the mask is
        symmetric across arms, so a right-arm-first layout is still correct; only a
        layout with different PER-ARM structure would break.
        """
        dims: list[int] = []
        for _ in range(self.arms):
            dims.append(self.joints_per_arm)
            if self.gripper_per_arm:
                dims.append(-1)
        return _transforms.make_bool_mask(*dims)

    def repack_map(self) -> dict[str, str]:
        """LEFT = intermediate key `RobotInputs` reads, RIGHT = LeRobot feature name.

        `RepackTransform` DISCARDS any key it does not list, which is the mechanism
        for dropping a camera: leave it out of `cameras` and it never appears here.
        """
        mapping = {image_key(slot): feature for slot, feature in self.real_cameras}
        mapping[STATE_KEY] = self.state_feature
        mapping[ACTIONS_KEY] = self.action_feature
        # `prompt_from_task` adds "prompt" BEFORE this repack runs; it must be listed
        # or RepackTransform drops it and the tokenizer raises "Prompt is required".
        mapping[PROMPT_KEY] = PROMPT_KEY
        return mapping


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:  # (c, h, w) -> (h, w, c)
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RobotInputs(_transforms.DataTransformFn):
    """Repacked LeRobot sample -> the nested dict the model expects."""

    spec: RobotSpec
    # PI0/PI05 mask the padding slot; PI0_FAST does not.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        real = {slot: _parse_image(data[image_key(slot)]) for slot, _ in self.spec.real_cameras}
        # Padding slots send zeros shaped like a real image; the model always expects
        # every entry of _model.IMAGE_KEYS to be present.
        reference = next(iter(real.values()))
        mask_padding = self.model_type != _model.ModelType.PI0_FAST

        images: dict[str, np.ndarray] = {}
        image_masks: dict[str, np.bool_] = {}
        for slot in _model.IMAGE_KEYS:
            if slot in real:
                images[slot] = real[slot]
                image_masks[slot] = np.True_
            else:
                images[slot] = np.zeros_like(reference)
                image_masks[slot] = np.False_ if mask_padding else np.True_

        inputs = {
            "state": data[STATE_KEY],
            "image": images,
            "image_mask": image_masks,
        }
        if ACTIONS_KEY in data:
            inputs[ACTIONS_KEY] = data[ACTIONS_KEY]
        if PROMPT_KEY in data:
            inputs[PROMPT_KEY] = data[PROMPT_KEY]
        return inputs


@dataclasses.dataclass(frozen=True)
class RobotOutputs(_transforms.DataTransformFn):
    """Model actions -> the robot's real action width.

    The slice is what makes a padded model `action_dim` safe: pi0.5 must run
    `action_dim=32` to match `pi05_base`'s projection shapes, and this trims the
    padding back off.
    """

    spec: RobotSpec

    def __call__(self, data: dict) -> dict:
        return {ACTIONS_KEY: np.asarray(data[ACTIONS_KEY][..., : self.spec.action_dim])}


def make_example(spec: RobotSpec) -> dict:
    """Random input example for dummy inference against `spec`."""
    example: dict = {
        image_key(slot): np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)
        for slot, _ in spec.real_cameras
    }
    example[STATE_KEY] = np.random.rand(spec.action_dim)
    example[PROMPT_KEY] = "do something"
    return example

"""Tests for the embodiment-parameterised policy transforms.

The important ones are the EQUIVALENCE tests. Generalising `yam_policy` and
`piper_policy` into one spec-driven implementation changes the objects that appear in
a `DataConfig`, so a structural before/after diff of the configs cannot prove the
behaviour is unchanged. Instead the OLD implementations are pinned here verbatim as
reference oracles, and the new pipeline is asserted to produce identical output for
identical input.

If you ever change `RobotInputs`/`RobotOutputs` on purpose, these tests are what tell
you which trained checkpoints you just invalidated.
"""

import dataclasses

import numpy as np
import pytest

from openpi.models import model as _model
from openpi.policies import robot_policy
import openpi.transforms as _transforms

# ---------------------------------------------------------------------------
# Reference oracles: the pre-refactor yam_policy / piper_policy behaviour.
# Copied verbatim so these tests do not depend on the deleted modules.
# ---------------------------------------------------------------------------

_YAM_REPACK = {
    "observation/top_image": "observation.images.top",
    "observation/left_wrist_image": "observation.images.left_wrist",
    "observation/right_wrist_image": "observation.images.right_wrist",
    "observation/state": "observation.state",
    "actions": "action",
    "prompt": "prompt",
}

_PIPER_REPACK = {
    "observation/front_image": "observation.images.front",
    "observation/right_image": "observation.images.right",
    # "observation.images.top" intentionally omitted.
    "observation/state": "observation.state",
    "actions": "action",
    "prompt": "prompt",
}


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    return image


def _legacy_yam_inputs(data: dict) -> dict:
    inputs = {
        "state": data["observation/state"],
        "image": {
            "base_0_rgb": _parse_image(data["observation/top_image"]),
            "left_wrist_0_rgb": _parse_image(data["observation/left_wrist_image"]),
            "right_wrist_0_rgb": _parse_image(data["observation/right_wrist_image"]),
        },
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


def _legacy_piper_inputs(data: dict, model_type: _model.ModelType) -> dict:
    scene = _parse_image(data["observation/front_image"])
    wrist = _parse_image(data["observation/right_image"])
    inputs = {
        "state": data["observation/state"],
        "image": {
            "base_0_rgb": scene,
            "left_wrist_0_rgb": wrist,
            "right_wrist_0_rgb": np.zeros_like(scene),
        },
        "image_mask": {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_ if model_type == _model.ModelType.PI0_FAST else np.False_,
        },
    }
    if "actions" in data:
        inputs["actions"] = data["actions"]
    if "prompt" in data:
        inputs["prompt"] = data["prompt"]
    return inputs


# ---------------------------------------------------------------------------
# Specs under test (kept local so the test does not depend on configs/).
# ---------------------------------------------------------------------------

YAM = robot_policy.RobotSpec(
    name="yam",
    cameras=(
        "observation.images.top",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    ),
)

PIPER_H = robot_policy.RobotSpec(
    name="piper_h",
    cameras=("observation.images.front", "observation.images.right", None),
)


def _raw_sample(features: list[str], *, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    sample: dict = {f: rng.integers(256, size=(224, 224, 3), dtype=np.uint8) for f in features}
    sample["observation.state"] = rng.random(14)
    sample["action"] = rng.random((50, 14))
    sample["prompt"] = "put the screwdriver in the bin"
    return sample


def _assert_same(new: dict, old: dict) -> None:
    assert new.keys() == old.keys()
    assert new["image"].keys() == old["image"].keys()
    assert new["image_mask"].keys() == old["image_mask"].keys()
    for slot in old["image"]:
        np.testing.assert_array_equal(new["image"][slot], old["image"][slot], err_msg=f"image[{slot}]")
        assert bool(new["image_mask"][slot]) == bool(old["image_mask"][slot]), f"mask[{slot}]"
    np.testing.assert_array_equal(new["state"], old["state"])
    np.testing.assert_array_equal(new["actions"], old["actions"])
    assert new["prompt"] == old["prompt"]


@pytest.mark.parametrize(
    "model_type", [_model.ModelType.PI0, _model.ModelType.PI05, _model.ModelType.PI0_FAST]
)
def test_yam_pipeline_matches_the_legacy_yam_policy(model_type):
    raw = _raw_sample(list(_YAM_REPACK.values())[:3] + ["observation.state", "action", "prompt"])

    new = robot_policy.RobotInputs(spec=YAM, model_type=model_type)(
        _transforms.RepackTransform(YAM.repack_map())(raw)
    )
    old = _legacy_yam_inputs(_transforms.RepackTransform(_YAM_REPACK)(raw))

    _assert_same(new, old)
    # All three YAM cameras are real, so nothing is ever masked.
    assert all(bool(m) for m in new["image_mask"].values())


@pytest.mark.parametrize(
    "model_type", [_model.ModelType.PI0, _model.ModelType.PI05, _model.ModelType.PI0_FAST]
)
def test_piper_pipeline_matches_the_legacy_piper_policy(model_type):
    raw = _raw_sample(
        [
            "observation.images.front",
            "observation.images.right",
            "observation.images.top",  # present in the dataset, must be dropped
            "observation.state",
            "action",
            "prompt",
        ]
    )

    new = robot_policy.RobotInputs(spec=PIPER_H, model_type=model_type)(
        _transforms.RepackTransform(PIPER_H.repack_map())(raw)
    )
    old = _legacy_piper_inputs(_transforms.RepackTransform(_PIPER_REPACK)(raw), model_type)

    _assert_same(new, old)


def test_the_dropped_camera_never_reaches_the_transform():
    raw = _raw_sample(
        ["observation.images.front", "observation.images.right", "observation.images.top",
         "observation.state", "action", "prompt"]
    )
    repacked = _transforms.RepackTransform(PIPER_H.repack_map())(raw)
    assert not any("top" in k for k in repacked), repacked.keys()


def test_padding_slot_is_zeros_and_masked_only_for_non_fast():
    raw = _raw_sample(["observation.images.front", "observation.images.right", "observation.state", "action", "prompt"])
    repacked = _transforms.RepackTransform(PIPER_H.repack_map())(raw)

    for model_type, expected_mask in [
        (_model.ModelType.PI0, False),
        (_model.ModelType.PI05, False),
        (_model.ModelType.PI0_FAST, True),
    ]:
        out = robot_policy.RobotInputs(spec=PIPER_H, model_type=model_type)(repacked)
        assert np.all(out["image"]["right_wrist_0_rgb"] == 0)
        assert out["image"]["right_wrist_0_rgb"].shape == out["image"]["base_0_rgb"].shape
        assert bool(out["image_mask"]["right_wrist_0_rgb"]) is expected_mask


def test_outputs_slice_to_the_spec_action_width():
    # The model runs action_dim=32 on pi0.5; the slice trims the padding back off.
    padded = {"actions": np.arange(32 * 2, dtype=np.float32).reshape(2, 32)}
    out = robot_policy.RobotOutputs(spec=YAM)(padded)
    assert out["actions"].shape == (2, 14)
    np.testing.assert_array_equal(out["actions"], padded["actions"][..., :14])


def test_delta_mask_matches_the_hand_written_one():
    assert YAM.delta_action_mask() == _transforms.make_bool_mask(6, -1, 6, -1)
    assert PIPER_H.delta_action_mask() == _transforms.make_bool_mask(6, -1, 6, -1)


def test_action_dim_is_derived_from_the_joint_layout():
    assert YAM.action_dim == 14
    single = robot_policy.RobotSpec(name="single", cameras=("cam", None, None), arms=1, joints_per_arm=6)
    assert single.action_dim == 7
    assert single.delta_action_mask() == _transforms.make_bool_mask(6, -1)
    no_gripper = robot_policy.RobotSpec(name="ng", cameras=("cam", None, None), arms=1, joints_per_arm=7,
                                        gripper_per_arm=False)
    assert no_gripper.action_dim == 7
    assert no_gripper.delta_action_mask() == _transforms.make_bool_mask(7)


def test_repack_map_lists_prompt_state_and_actions():
    mapping = YAM.repack_map()
    assert mapping["prompt"] == "prompt"  # or TokenizePrompt raises "Prompt is required"
    assert mapping["observation/state"] == "observation.state"
    assert mapping["actions"] == "action"


def test_a_gap_in_the_middle_of_the_slots_is_rejected():
    """Real cameras must fill leading slots; openpi's policies only mask the trailing one."""
    with pytest.raises(ValueError, match="real cameras must fill the leading slots"):
        robot_policy.RobotSpec(name="bad", cameras=("cam_a", None, "cam_b"))


def test_wrong_number_of_slots_is_rejected():
    with pytest.raises(ValueError, match="exactly 3 entries"):
        robot_policy.RobotSpec(name="bad", cameras=("cam_a", "cam_b"))


def test_a_spec_with_no_real_camera_is_rejected():
    with pytest.raises(ValueError, match="at least one camera slot"):
        robot_policy.RobotSpec(name="bad", cameras=(None, None, None))


def test_spec_is_hashable_and_frozen():
    # It lives inside a frozen DataConfigFactory dataclass, so it must be both.
    assert hash(YAM) == hash(dataclasses.replace(YAM))
    with pytest.raises(dataclasses.FrozenInstanceError):
        YAM.name = "nope"  # type: ignore[misc]

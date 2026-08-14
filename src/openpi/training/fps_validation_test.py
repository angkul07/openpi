"""Tests for control-frequency validation in the data loader.

Nothing in openpi models time: `action_horizon` counts STEPS, and `delta_timestamps`
is built from each source's OWN fps. So a mixture of 20 Hz and 30 Hz data trains on
two different notions of "the next 50 steps" and two different delta-action scales,
normalised as one distribution, with nothing in the loss to show for it. These are the
checks that make that loud.

Kept free of heavy dataset construction: `_assert_uniform_fps` and `RobotSpec` are
pure, so they can be tested directly.
"""

import dataclasses

import pytest

from openpi.policies.robot_policy import RobotSpec
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def _source(repo_id: str) -> _config.MixtureSource:
    return _config.MixtureSource(repo_id=repo_id, samples_per_batch=32)


def test_uniform_fps_passes():
    sources = [_source("org/teleop"), _source("org/ego")]
    _data_loader._assert_uniform_fps(sources, [30.0, 30.0], 50)  # noqa: SLF001


def test_single_source_passes():
    _data_loader._assert_uniform_fps([_source("org/only")], [20.0], 50)  # noqa: SLF001


def test_mixed_fps_raises_and_reports_both_durations():
    sources = [_source("org/teleop_20hz"), _source("org/ego_30hz")]
    with pytest.raises(ValueError, match="different control frequencies") as exc:
        _data_loader._assert_uniform_fps(sources, [20.0, 30.0], 50)  # noqa: SLF001

    message = str(exc.value)
    assert "different control frequencies" in message
    # The consequence must be spelled out in seconds, not just in fps.
    assert "2.50 s per chunk" in message
    assert "1.67 s per chunk" in message
    assert "org/teleop_20hz" in message
    assert "org/ego_30hz" in message


# ---------------------------------------------------------------------------
# RobotSpec.control_hz
# ---------------------------------------------------------------------------

_SPEC = RobotSpec(name="t", cameras=("cam", None, None), control_hz=30.0)


def test_chunk_duration_is_derived():
    assert _SPEC.chunk_duration_s(50) == pytest.approx(50 / 30)
    # The same action_horizon means something different at another rate -- the whole
    # reason the constant is declared.
    assert dataclasses.replace(_SPEC, control_hz=20.0).chunk_duration_s(50) == pytest.approx(2.5)


def test_chunk_duration_is_none_when_undeclared():
    assert dataclasses.replace(_SPEC, control_hz=None).chunk_duration_s(50) is None


def test_non_positive_control_hz_is_rejected():
    with pytest.raises(ValueError, match="control_hz must be positive"):
        RobotSpec(name="t", cameras=("cam", None, None), control_hz=0)


def test_expected_fps_reaches_the_data_config():
    """The spec's declared rate is what the loader checks each dataset against."""
    import openpi.models.pi0_config as pi0_config

    factory = _config.LeRobotRobotDataConfig(robot=_SPEC, repo_id="org/x")
    created = factory.create(_config.pathlib.Path("/tmp/assets"), pi0_config.Pi0Config())
    assert created.expected_fps == 30.0


def test_expected_fps_is_none_when_the_spec_does_not_declare_one():
    import openpi.models.pi0_config as pi0_config

    factory = _config.LeRobotRobotDataConfig(
        robot=dataclasses.replace(_SPEC, control_hz=None), repo_id="org/x"
    )
    created = factory.create(_config.pathlib.Path("/tmp/assets"), pi0_config.Pi0Config())
    assert created.expected_fps is None


def test_the_shipped_specs_declare_their_measured_rates():
    from configs._shared.robots import PIPER_H
    from configs._shared.robots import YAM

    assert YAM.control_hz == 30.0
    assert PIPER_H.control_hz == 20.0
    # 50 steps is a different amount of future on each -- they are not comparable at
    # equal action_horizon even though the number looks the same.
    assert YAM.chunk_duration_s(50) == pytest.approx(1.667, abs=1e-3)
    assert PIPER_H.chunk_duration_s(50) == pytest.approx(2.5)


def test_near_identical_rates_are_not_flagged():
    """29.97 and 30.0 are the same rate for this purpose; flagging them would be noise."""
    sources = [_source("org/a"), _source("org/b")]
    _data_loader._assert_uniform_fps(sources, [29.97, 30.0], 50)  # noqa: SLF001

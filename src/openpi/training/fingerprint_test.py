"""Tests for norm-stat fingerprinting.

Deliberately free of any jax/openpi-model import so this runs in about a second.
"""

import dataclasses
import json
import pathlib

import pytest

from openpi.training import fingerprint


@dataclasses.dataclass(frozen=True)
class _Source:
    """Stand-in for MixtureSource -- fingerprinting is structural, not nominal."""

    repo_id: str
    samples_per_batch: int
    root: str | None = None
    exclude_episodes: tuple[int, ...] = ()
    holdout_fraction: float = 0.0
    holdout_seed: int = 0


def _sources(teleop_per_batch: int = 32, **kwargs) -> tuple[_Source, ...]:
    return (
        _Source("org/teleop", teleop_per_batch, root="/data/teleop", **kwargs),
        _Source("org/ego", 64 - teleop_per_batch, root="/data/ego"),
    )


def test_no_sources_has_no_fingerprint():
    assert fingerprint.compute(()) is None
    assert fingerprint.compute(None) is None


def test_digest_is_stable_and_batch_size_is_derived():
    fp = fingerprint.compute(_sources())
    assert fp is not None
    assert fp["batch_size"] == 64
    assert fp["digest"] == fingerprint.compute(_sources())["digest"]


@pytest.mark.parametrize(
    "changed",
    [
        pytest.param({"teleop_per_batch": 24}, id="draw"),
        pytest.param({"exclude_episodes": (1, 2, 3)}, id="holdout"),
        pytest.param({"holdout_fraction": 0.1}, id="holdout_fraction"),
    ],
)
def test_any_distribution_change_changes_the_digest(changed):
    base = fingerprint.compute(_sources())["digest"]
    assert fingerprint.compute(_sources(**changed))["digest"] != base


def test_source_order_is_part_of_identity():
    # The sampler concatenates in order and the draw counts are positional.
    forward = fingerprint.compute(_sources())
    reversed_ = fingerprint.compute(tuple(reversed(_sources())))
    assert forward["digest"] != reversed_["digest"]


def test_roundtrip_write_read_check(tmp_path: pathlib.Path):
    fp = fingerprint.compute(_sources())
    fingerprint.write(tmp_path, fp, config_name="arm_a", asset_id="p50")

    written = json.loads((tmp_path / fingerprint.FINGERPRINT_FILE).read_text())
    assert written["written_by_config"] == "arm_a"
    assert written["asset_id"] == "p50"

    # Matching config: no raise.
    fingerprint.check(tmp_path, fp, config_name="arm_a", asset_id="p50")


def test_check_raises_when_stats_came_from_a_different_mixture(tmp_path: pathlib.Path):
    """The `cp -r` from a neighbouring arm. This is the whole reason the module exists."""
    fingerprint.write(tmp_path, fingerprint.compute(_sources(32)), config_name="arm_a", asset_id="p50")

    with pytest.raises(ValueError, match="NOT computed for config"):
        fingerprint.check(tmp_path, fingerprint.compute(_sources(24)), config_name="arm_b", asset_id="p375")


def test_legacy_stats_without_a_fingerprint_warn_but_load(tmp_path: pathlib.Path, caplog):
    """Boxes that already have assets must keep working."""
    with caplog.at_level("WARNING"):
        fingerprint.check(tmp_path, fingerprint.compute(_sources()), config_name="arm_a", asset_id="p50")
    assert "no fingerprint" in caplog.text


def test_check_is_a_noop_for_non_mixture_configs(tmp_path: pathlib.Path):
    fingerprint.check(tmp_path, None, config_name="pi05_libero", asset_id="libero")


def test_unreadable_fingerprint_raises(tmp_path: pathlib.Path):
    (tmp_path / fingerprint.FINGERPRINT_FILE).write_text("{not json")
    with pytest.raises(ValueError, match="unreadable"):
        fingerprint.read(tmp_path)

"""Tests for out-of-tree config discovery.

These import `openpi.training.config`, which is slow (jax), so they are deliberately
few and each one covers a failure that would otherwise be silent or confusing.
"""

import dataclasses
import os
import pathlib
import textwrap

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from openpi.training import config as _config
from openpi.training import registry


def test_the_fd_arms_are_discovered():
    names = _config.config_names()
    for expected in ("pi05_yam7h_ea", "pi05_50run_ea", "pi05_abcego_sd", "pi0_fast_yam_mix_ea"):
        assert expected in names, f"{expected} missing -- discovery did not reach configs/fd/"


def test_builtin_openpi_configs_still_resolve():
    assert _config.get_config("debug").name == "debug"
    assert _config.get_config("pi05_libero").name == "pi05_libero"


def test_client_configs_are_owned_by_their_module():
    assert registry.owner("pi05_50run_ea") == "configs.fd.yam.run50"
    # A built-in is not a client config.
    assert registry.owner("pi05_libero") is None


def test_unknown_config_still_suggests_a_near_match():
    with pytest.raises(ValueError, match="Did you mean 'pi05_yam7h_ea'"):
        _config.get_config("pi05_yam7h_eaa")


def test_duplicate_registration_names_both_owners():
    existing = _config.get_config("pi05_50run_ea")
    with pytest.raises(registry.ConfigDiscoveryError, match="duplicate config name"):
        registry.register(dataclasses.replace(existing))


def test_registering_something_without_a_name_raises():
    with pytest.raises(registry.ConfigDiscoveryError, match="has no `name`"):
        registry.register(object())


def test_configs_root_honours_the_env_override(tmp_path: pathlib.Path, monkeypatch):
    monkeypatch.setenv("OPENPI_CONFIGS_DIR", str(tmp_path))
    assert registry.configs_root() == tmp_path.resolve()


def test_a_missing_configs_dir_is_a_loud_error_not_a_silent_skip(tmp_path: pathlib.Path, monkeypatch):
    monkeypatch.setenv("OPENPI_CONFIGS_DIR", str(tmp_path / "nope"))
    with pytest.raises(registry.ConfigDiscoveryError, match="is not a directory"):
        registry.configs_root()


def test_underscore_prefixed_paths_are_not_scanned(tmp_path: pathlib.Path):
    (tmp_path / "_shared").mkdir()
    (tmp_path / "_shared" / "helpers.py").touch()
    (tmp_path / "client").mkdir()
    (tmp_path / "client" / "arm.py").touch()
    (tmp_path / "client" / "_private.py").touch()

    found = registry._iter_modules(tmp_path)  # noqa: SLF001
    assert found == ["configs.client.arm"]


def test_a_broken_config_module_raises_rather_than_vanishing(tmp_path: pathlib.Path, monkeypatch):
    """A config that fails to import must never be silently dropped from the CLI."""
    root = tmp_path / "configs"
    (root / "broken").mkdir(parents=True)
    (root / "__init__.py").touch()
    (root / "broken" / "__init__.py").touch()
    (root / "broken" / "arm.py").write_text(textwrap.dedent("raise RuntimeError('boom')"))

    monkeypatch.setenv("OPENPI_CONFIGS_DIR", str(root))
    monkeypatch.syspath_prepend(str(tmp_path))
    # reset() also evicts the real `configs` package from sys.modules, which is what
    # lets the temporary one be imported in its place.
    registry.reset()
    try:
        with pytest.raises(registry.ConfigDiscoveryError, match="failed to import config module"):
            registry.discover()
    finally:
        # Put the real registry back for whatever runs next.
        monkeypatch.undo()
        registry.reset()
        registry.discover()
    assert "pi05_50run_ea" in registry.registered()

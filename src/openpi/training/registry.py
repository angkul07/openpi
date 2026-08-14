"""Discovery and registration for out-of-tree training configs.

Experiment arms do not live in `config.py`. They live in `configs/<client>/...` and
register themselves here, so that:

  * `config.py` stays an upstream file we barely touch, and rebasing onto upstream
    openpi stops conflicting on a 900-line list of our own experiments;
  * one client's arms cannot silently collide with another's;
  * an arm that fails to import is a loud error, not a config that quietly vanishes
    from the CLI.

A client module registers at import time::

    from openpi.training import registry
    registry.register(TrainConfig(name="mm_pi05_ea", ...))

Discovery is lazy: it runs the first time anyone asks for a config by name (see
`config.get_config`), by which point `config.py` has finished importing and client
modules can import freely from it.

Layout under the configs root::

    configs/
      _shared/          # helpers -- leading underscore, never scanned for arms
      _template/        # skeleton to copy for a new client
      fd/               # one directory per client
        datasets.py     # that client's roots, repo ids, holdout indices
        yam/
          yam7h.py      # arms, each calling register()

Names starting with `_` are skipped by the scanner but remain importable, which is
how `_shared` and `_template` stay out of the registry while still being usable.

The configs root is resolved in this order, first hit wins:
  1. `$OPENPI_CONFIGS_DIR`
  2. `<repo root>/configs`, relative to this file
  3. `$CWD/configs`
If none exists, discovery is a silent no-op -- a plain upstream checkout keeps working.
"""

from __future__ import annotations

import importlib
import logging
import os
import pathlib
import sys
from typing import Any

logger = logging.getLogger(__name__)

# Name of the importable package that holds client configs.
_PACKAGE = "configs"

# name -> (config, owning module) so collisions can name both culprits.
_REGISTERED: dict[str, tuple[Any, str]] = {}
_DISCOVERED = False
# Set while discovery is running, so a client module that calls get_config() at import
# time gets a clear error instead of infinite recursion.
_DISCOVERING = False


class ConfigDiscoveryError(RuntimeError):
    """Raised when a client config module cannot be imported or registered."""


def configs_root() -> pathlib.Path | None:
    """Locate the configs directory, or None if this checkout has none."""
    if env := os.environ.get("OPENPI_CONFIGS_DIR"):
        root = pathlib.Path(env).expanduser().resolve()
        if not root.is_dir():
            raise ConfigDiscoveryError(f"OPENPI_CONFIGS_DIR={env!r} is not a directory")
        return root

    # <repo>/src/openpi/training/registry.py -> <repo>
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    for candidate in (repo_root / _PACKAGE, pathlib.Path.cwd() / _PACKAGE):
        if candidate.is_dir():
            return candidate.resolve()
    return None


def register(*configs: Any) -> None:
    """Register one or more TrainConfigs from a client module.

    Raises on a duplicate name, naming both the existing owner and the new one. Two
    clients reaching for the same arm name is the exact accident this prevents.
    """
    # The caller's module name, for error messages and provenance.
    frame = sys._getframe(1)  # noqa: SLF001
    origin = frame.f_globals.get("__name__", "<unknown>")

    for config in configs:
        name = getattr(config, "name", None)
        if not name:
            raise ConfigDiscoveryError(f"{origin}: registered object has no `name` ({config!r})")
        if name in _REGISTERED:
            _, existing = _REGISTERED[name]
            raise ConfigDiscoveryError(
                f"duplicate config name {name!r}: already registered by {existing}, now also by {origin}. "
                "Config names are global; prefix client arms with the client slug (e.g. 'mm_pi05_ea')."
            )
        _REGISTERED[name] = (config, origin)


def registered() -> dict[str, Any]:
    """All discovered client configs, keyed by name. Triggers discovery."""
    discover()
    return {name: config for name, (config, _) in _REGISTERED.items()}


def owner(name: str) -> str | None:
    """Which module registered `name`, or None if it is not a client config."""
    discover()
    entry = _REGISTERED.get(name)
    return entry[1] if entry else None


def _iter_modules(root: pathlib.Path) -> list[str]:
    """Dotted module names for every arm file under `root`, deterministically ordered.

    Skips anything whose path contains a component starting with `_` or `.`, which is
    what keeps `_shared`, `_template` and `__pycache__` out of the registry.
    """
    modules: list[str] = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        parts = rel.with_suffix("").parts
        if any(part.startswith(("_", ".")) for part in parts):
            continue
        modules.append(".".join((_PACKAGE, *parts)))
    return modules


def discover(*, force: bool = False) -> None:
    """Import every client config module exactly once. Idempotent."""
    global _DISCOVERED, _DISCOVERING  # noqa: PLW0603

    if _DISCOVERING:
        raise ConfigDiscoveryError(
            "get_config() was called while client configs were still being discovered. "
            "A config module must not look up other configs by name at import time."
        )
    if _DISCOVERED and not force:
        return

    root = configs_root()
    if root is None:
        _DISCOVERED = True
        return

    # Make `import configs.<client>...` work, and make sure it is OUR configs package
    # and not some unrelated one that happens to be importable first.
    parent = str(root.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    if (existing := sys.modules.get(_PACKAGE)) is not None:
        found = pathlib.Path(getattr(existing, "__file__", "") or "").resolve().parent
        if found != root:
            raise ConfigDiscoveryError(
                f"a different {_PACKAGE!r} package is already imported from {found}, "
                f"shadowing the configs root at {root}. Set OPENPI_CONFIGS_DIR or fix sys.path."
            )

    _DISCOVERING = True
    try:
        modules = _iter_modules(root)
        for module_name in modules:
            try:
                if force and module_name in sys.modules:
                    importlib.reload(sys.modules[module_name])
                else:
                    importlib.import_module(module_name)
            except ConfigDiscoveryError:
                raise
            except Exception as e:
                # Never swallow this. A config that fails to import must be a hard
                # error -- silently dropping it is how you launch the wrong arm.
                raise ConfigDiscoveryError(f"failed to import config module {module_name}: {e}") from e
    finally:
        _DISCOVERING = False

    _DISCOVERED = True
    logger.debug("discovered %d configs from %s", len(_REGISTERED), root)


def reset() -> None:
    """Forget everything discovered so far. For tests only.

    Also drops the client modules from `sys.modules`. Without that, a later `discover()`
    would import them from cache, skip their module bodies, and therefore never re-run
    `register()` -- leaving the registry permanently empty.
    """
    global _DISCOVERED  # noqa: PLW0603
    _REGISTERED.clear()
    _DISCOVERED = False
    for name in [n for n in sys.modules if n == _PACKAGE or n.startswith(f"{_PACKAGE}.")]:
        del sys.modules[name]

"""Fingerprints that bind norm stats to the distribution they were computed from.

The failure this exists to prevent, in full:

  Norm stats for a mixture are computed *through the training sampler*, so they
  describe the sampled distribution (say 24 teleop / 40 ego), not what is on disk.
  Change the draw, the roots, or the holdout, and the old stats are wrong. But
  `run_yam.sh` skips stage [1/3] whenever `norm_stats.json` already exists, and a
  stats directory is just a directory -- so a `cp -r` from a neighbouring arm, or an
  `asset_id` reused across two mixtures, silently normalises training against the
  wrong distribution. Nothing errors. The run completes. The numbers are quietly junk.

So: whoever writes norm stats also writes a fingerprint of the distribution, and
whoever loads them checks it. A mismatch raises instead of training.

Stats written before this existed have no fingerprint file. Those load with a warning
rather than an error -- refusing them would strand every box that already has assets.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import pathlib
from typing import Any

logger = logging.getLogger(__name__)

FINGERPRINT_FILE = "norm_stats_fingerprint.json"

# Bump when the fingerprint's meaning changes. An old fingerprint against a new
# version warns (it cannot be compared) rather than failing.
VERSION = 1


def _source_entry(source: Any) -> dict[str, Any]:
    """Canonical description of one MixtureSource.

    Only the fields that change the sampled distribution go in. `exclude_episodes` is
    reduced to a count plus a digest: the 249-element holdout would otherwise dominate
    the file, but a changed holdout must still register as a change.
    """
    excluded = tuple(int(e) for e in getattr(source, "exclude_episodes", ()) or ())
    return {
        "repo_id": source.repo_id,
        "root": source.root,
        "samples_per_batch": int(source.samples_per_batch),
        "exclude_episodes_count": len(excluded),
        "exclude_episodes_digest": hashlib.sha256(json.dumps(sorted(excluded)).encode()).hexdigest()[:16]
        if excluded
        else None,
        "holdout_fraction": float(getattr(source, "holdout_fraction", 0.0) or 0.0),
        "holdout_seed": int(getattr(source, "holdout_seed", 0) or 0),
    }


def compute(sources: Any) -> dict[str, Any] | None:
    """Fingerprint a mixture. Returns None if there is nothing to fingerprint."""
    sources = tuple(sources or ())
    if not sources:
        return None
    # Source ORDER is part of the identity: the sampler concatenates in this order and
    # the per-source draw counts are positional.
    entries = [_source_entry(s) for s in sources]
    body = {
        "version": VERSION,
        "batch_size": sum(e["samples_per_batch"] for e in entries),
        "sources": entries,
    }
    body["digest"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
    return body


def write(directory: pathlib.Path | str, fp: dict[str, Any] | None, *, config_name: str, asset_id: str) -> None:
    """Write the fingerprint next to norm_stats.json. No-op when there is none."""
    if fp is None:
        return
    path = pathlib.Path(directory) / FINGERPRINT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**fp, "written_by_config": config_name, "asset_id": asset_id}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    logger.info("wrote norm-stats fingerprint %s to %s", fp["digest"], path)


def read(directory: pathlib.Path | str) -> dict[str, Any] | None:
    path = pathlib.Path(directory) / FINGERPRINT_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"norm-stats fingerprint at {path} is unreadable: {e}") from e


def _describe(fp: dict[str, Any]) -> str:
    lines = [f"  batch_size={fp.get('batch_size')} digest={fp.get('digest')}"]
    lines.extend(
        f"    {s.get('repo_id')} x{s.get('samples_per_batch')} root={s.get('root')} "
        f"excluded={s.get('exclude_episodes_count')}"
        for s in fp.get("sources", [])
    )
    return "\n".join(lines)


def check(directory: pathlib.Path | str, expected: dict[str, Any] | None, *, config_name: str, asset_id: str) -> None:
    """Verify stats in `directory` were computed for `expected`. Raise if not."""
    if expected is None:
        return

    found = read(directory)
    if found is None:
        logger.warning(
            "norm stats at %s have no fingerprint (written before fingerprinting, or copied "
            "from another arm). Cannot verify they match config %r. Recompute them to silence "
            "this: uv run scripts/compute_norm_stats.py --config-name %s --max-frames 200000 --skip-videos",
            directory,
            config_name,
            config_name,
        )
        return

    if found.get("version") != expected["version"]:
        logger.warning(
            "norm-stats fingerprint at %s is version %s, this openpi writes version %s -- not comparable. "
            "Recompute to verify.",
            directory,
            found.get("version"),
            expected["version"],
        )
        return

    if found.get("digest") == expected["digest"]:
        return

    raise ValueError(
        f"norm stats at {directory} were NOT computed for config {config_name!r} (asset_id {asset_id!r}).\n"
        f"Training against them would normalise the wrong distribution.\n"
        f"stats were written for (by config {found.get('written_by_config')!r}):\n{_describe(found)}\n"
        f"this config needs:\n{_describe(expected)}\n"
        f"Fix: delete that directory and recompute --\n"
        f"  rm -rf {directory}\n"
        f"  uv run scripts/compute_norm_stats.py --config-name {config_name} --max-frames 200000 --skip-videos"
    )


def from_data_config_factory(factory: Any) -> dict[str, Any] | None:
    """Fingerprint whatever mixture a data-config factory declares, if any."""
    sources = getattr(factory, "sources", None)
    if not sources:
        return None
    if not all(dataclasses.is_dataclass(s) for s in sources):
        return None
    return compute(sources)

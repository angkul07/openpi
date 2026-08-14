"""Dump a canonical, diffable representation of training configs.

This exists to prove that moving an experiment arm out of `config.py` and into
`configs/<client>/...` changed nothing. Dump before the move, dump after, diff the
two files: an empty diff is the whole acceptance test.

    uv run python scripts/dump_configs.py --out /tmp/before.json
    # ... migrate ...
    uv run python scripts/dump_configs.py --out /tmp/after.json
    diff /tmp/before.json /tmp/after.json

Canonicalisation notes:

  * Dataclasses recurse field-by-field, so field ORDER never affects the output and
    a reordered constructor call still compares equal.
  * Anything that is not a dataclass/dict/sequence/scalar falls back to `repr()`.
    That covers `freeze_filter` (an nnx filter tree), weight loaders and transforms
    -- all of which have stable, structural reprs.
  * `data` is dumped twice: the FACTORY (what the config literal says) and the
    CREATED `DataConfig` (what training actually consumes, transforms included).
    The created form is the one that catches a transform that silently went missing.
  * Norm stats are reduced to a presence flag. They live outside the repo, differ
    per box, and would otherwise swamp the diff.
"""

import argparse
import dataclasses
import json
import pathlib
import re
from typing import Any

import openpi.training.config as _config


def _scrub(text: str) -> str:
    """Strip per-process identity out of a repr.

    Objects without a structural repr fall back to `<Foo object at 0x7f...>`, and
    tyro's MISSING sentinel prints its `id()`. Both change every run, so without this
    every diff is 100% noise.
    """
    text = re.sub(r"0x[0-9a-fA-F]+", "0xPTR", text)
    return re.sub(r"id='\d+'", "id='ID'", text)


def _canon(value: Any, *, depth: int = 0) -> Any:
    """Recursively reduce a value to JSON-comparable form."""
    if depth > 12:  # cycles are not expected here, but never hang on one
        return f"<max-depth {type(value).__name__}>"
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "__type__": type(value).__name__,
            **{f.name: _canon(getattr(value, f.name), depth=depth + 1) for f in dataclasses.fields(value)},
        }
    if isinstance(value, dict):
        # Sort so dict ordering never shows up as a spurious diff.
        return {str(k): _canon(v, depth=depth + 1) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, list | tuple):
        return [_canon(v, depth=depth + 1) for v in value]
    if isinstance(value, pathlib.Path):
        return str(value)
    return _scrub(repr(value))


def _dump_one(name: str) -> dict[str, Any]:
    config = _config.get_config(name)
    out = _canon(config)

    # Also dump what `data.create()` actually produces -- the transforms, the mixture
    # and the resolved asset id. A migration that drops a transform would otherwise
    # pass, since the factory literal alone would still look identical.
    try:
        created = config.data.create(config.assets_dirs, config.model)
        created_canon = _canon(created)
        # Norm stats are per-box and enormous; keep only whether they were found.
        if isinstance(created_canon, dict):
            created_canon["norm_stats"] = "<present>" if created.norm_stats is not None else None
        out["__created_data_config__"] = created_canon
    except Exception as e:
        out["__created_data_config__"] = f"<create() raised: {type(e).__name__}: {e}>"

    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="Path to write the JSON dump to.")
    p.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="Config names to dump. Default: every registered config.",
    )
    args = p.parse_args()

    names = args.configs if args.configs else sorted(_config.config_names())
    dump = {name: _dump_one(name) for name in names}

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dump, indent=2, sort_keys=True))
    print(f"wrote {len(dump)} configs -> {out}")


if __name__ == "__main__":
    main()

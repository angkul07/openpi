"""Validate a re-indexed LeRobot v2.1 episode subset against the dataset it was cut from.

    uv run python vast_run/mm/verify_lerobot_subset.py <src_root> <dst_root>

Expected counts are derived from the SOURCE dataset's meta and from `meta/source.json`'s
`kept_source_episodes`, never from what the builder wrote into the destination -- so
these are genuine cross-checks rather than restatements. The two that matter most:

  * the global `index` column must be contiguous across the whole subset (LeRobot uses
    it as the flat row counter; a gap silently misaligns the dataset against its length)
  * every episode must still resolve to its ORIGINAL task sentence after task_index is
    renumbered (a wrong map feeds the model the wrong prompt under prompt_from_task,
    which surfaces as "the mixture underperforms" and never as an error)
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

VIDEO_KEYS = ("observation.images.front", "observation.images.wrist")


def _jl(path: pathlib.Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def verify(src: pathlib.Path, dst: pathlib.Path) -> list[str]:
    fails: list[str] = []

    def check(cond: bool, msg: str) -> None:
        print(("  OK   " if cond else "  FAIL ") + msg)
        if not cond:
            fails.append(msg)

    info = json.loads((dst / "meta/info.json").read_text())
    keep = json.loads((dst / "meta/source.json").read_text())["kept_source_episodes"]
    n = len(keep)

    src_eps = {e["episode_index"]: e for e in _jl(src / "meta/episodes.jsonl")}
    src_tasks = {t["task_index"]: t["task"] for t in _jl(src / "meta/tasks.jsonl")}
    expect_frames = sum(src_eps[o]["length"] for o in keep)

    print(f"== {dst.name}: {n} eps, expect {expect_frames} frames (from SOURCE meta) ==")
    check(info["codebase_version"] == "v2.1", "codebase_version v2.1")
    check(info["total_episodes"] == n, f"total_episodes {info['total_episodes']} == {n}")
    check(info["total_frames"] == expect_frames, f"total_frames {info['total_frames']} == {expect_frames}")
    check(info["total_videos"] == n * 2, f"total_videos {info['total_videos']} == {n * 2}")
    check(info["splits"] == {"train": f"0:{n}"}, f"splits {info['splits']}")
    check(info["fps"] == 30, "fps 30")
    check(len(set(keep)) == n, "no duplicate source episodes selected")

    eps = _jl(dst / "meta/episodes.jsonl")
    tasks = _jl(dst / "meta/tasks.jsonl")
    stats = _jl(dst / "meta/episodes_stats.jsonl")
    check([e["episode_index"] for e in eps] == list(range(n)), "episodes.jsonl contiguous 0..n-1")
    check([t["task_index"] for t in tasks] == list(range(len(tasks))), "tasks.jsonl contiguous")
    check(
        len(stats) == n and [s["episode_index"] for s in stats] == list(range(n)),
        f"episodes_stats.jsonl {len(stats)} rows, contiguous",
    )
    check(sum(e["length"] for e in eps) == info["total_frames"], "episode lengths sum to total_frames")
    check(info["total_tasks"] == len(tasks), f"total_tasks {info['total_tasks']} == {len(tasks)}")
    check(
        {t["task"] for t in tasks} <= set(src_tasks.values()),
        f"{len(tasks)} task strings all present in source's {len(src_tasks)}",
    )
    check(
        [e["length"] for e in eps] == [src_eps[o]["length"] for o in keep],
        "per-episode lengths match their source episodes in order",
    )

    pqs = sorted((dst / "data/chunk-000").glob("*.parquet"))
    check(len(pqs) == n, f"parquet files {len(pqs)} == {n}")
    for k in VIDEO_KEYS:
        check(len(list((dst / "videos/chunk-000" / k).glob("*.mp4"))) == n, f"{k}: {n} mp4")

    run = bad_idx = bad_task = bad_ep = 0
    ntasks = len(tasks)
    for new in range(n):
        df = pd.read_parquet(pqs[new])
        if df["index"].tolist() != list(range(run, run + len(df))):
            bad_idx += 1
        if (df["episode_index"] != new).any():
            bad_ep += 1
        if not df["task_index"].between(0, ntasks - 1).all():
            bad_task += 1
        run += len(df)
    check(bad_idx == 0, f"global `index` contiguous ({bad_idx} bad)")
    check(bad_ep == 0, f"episode_index stamped correctly ({bad_ep} bad)")
    check(bad_task == 0, f"task_index within [0,{ntasks - 1}] ({bad_task} bad)")
    check(run == expect_frames, f"rows walked {run} == {expect_frames}")

    rng = np.random.default_rng(0)
    mism = 0
    for new in rng.choice(n, min(12, n), replace=False):
        a = pd.read_parquet(dst / f"data/chunk-000/episode_{int(new):06d}.parquet")
        b = pd.read_parquet(src / f"data/chunk-000/episode_{keep[int(new)]:06d}.parquet")
        same = len(a) == len(b) and all(
            np.allclose(np.stack(list(a[c])), np.stack(list(b[c])))
            for c in ("observation.state", "action", "timestamp", "frame_index")
        )
        if not same:
            mism += 1
    check(mism == 0, f"12 random episodes byte-equal on state/action/timestamp/frame_index ({mism} bad)")

    idx2task = {t["task_index"]: t["task"] for t in tasks}
    bad = 0
    for new in range(n):
        df = pd.read_parquet(pqs[new])
        if {idx2task[int(i)] for i in df["task_index"].unique()} != set(src_eps[keep[new]]["tasks"]):
            bad += 1
    check(bad == 0, f"every episode resolves to its ORIGINAL task sentence ({bad} bad)")

    return fails


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    fails = verify(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
    print(f"\nFAILURES: {len(fails)}")
    for f in fails:
        print("  -", f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Download the episodes named in manifest.json and assemble two LeRobot v2.1 datasets.

Only the selected episodes are fetched -- 4,174 of 8,842 ego and 761 of 1,645 teleop --
so this pulls ~2 GiB instead of the ~4.3 GiB of the two full sources.

Runs in two phases:
  1. download  -- threaded hf_hub_download of each selected episode's parquet + 3 videos.
                  Files land in the HF cache; re-running is free for anything already there.
  2. assemble  -- renumber episodes 0..N-1 and rewrite meta. Sequential and deterministic.

Why renumber: lerobot's `get_episodes_file_paths` probes `range(total_episodes)` for local
files, so a folder holding sparse original indices (12, 20, 44, ...) sends it looking for
episode 0, the local check fails, and it silently falls back to the hub. The original
indices are preserved in `source_episodes.json` inside each output dataset.

`tasks.jsonl` is copied verbatim rather than compacted: parquet rows carry `task_index`
values pointing into it, and remapping them buys nothing but a chance to corrupt the
task mapping. Unused task rows are harmless.

Usage:
    python build_mixture.py --manifest manifest.json --out-root /workspace/data/yam7h
    python build_mixture.py --verify-only --out-root /workspace/data/yam7h
"""

import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import random
import shutil
import sys
import threading
import time

# Must be set BEFORE huggingface_hub is imported -- both are read into
# huggingface_hub.constants at import time.
#
# Xet is disabled deliberately. With hf_xet installed, hub>=1.x routes downloads
# through the xet transport, which wedged this build: the process sat on a single
# ESTABLISHED socket with no keepalive timer and no bytes moving for 23 minutes.
# A socket with no timer never times out, so neither the download nor the retry
# wrapper below can ever recover -- it hangs forever rather than failing. Plain
# HTTP honours HF_HUB_DOWNLOAD_TIMEOUT, so a stall surfaces as an exception that
# fetch() can retry.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

import pandas as pd  # noqa: E402
from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402

FPS = 30
_print_lock = threading.Lock()
_retries = [0]


def fetch(repo: str, rel: str, attempts: int = 8) -> str:
    """hf_hub_download with backoff.

    The Hub rate-limits hard (HTTP 429 on the xet-read-token endpoint) when many
    workers pull at once, and unauthenticated clients get a much lower ceiling --
    an unauthenticated 16-worker run failed 2062/3044 files. 429s are transient,
    so retry with exponential backoff + jitter rather than aborting the run.
    Anything already in the cache returns immediately and never reaches the network.
    """
    delay = 2.0
    for attempt in range(attempts):
        try:
            return hf_hub_download(repo, rel, repo_type="dataset")
        except Exception:  # noqa: BLE001 -- 429/timeout/connection reset are all retryable
            if attempt == attempts - 1:
                raise
            with _print_lock:
                _retries[0] += 1
            time.sleep(delay + random.uniform(0, delay))
            delay = min(delay * 2, 60.0)
    raise AssertionError("unreachable")


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def episode_files(info: dict, episode: int) -> list[str]:
    """Repo-relative paths for one episode: its parquet plus one video per camera."""
    chunk = episode // info["chunks_size"]
    out = [info["data_path"].format(episode_chunk=chunk, episode_index=episode)]
    for key, feat in info["features"].items():
        if feat["dtype"] == "video":
            out.append(info["video_path"].format(
                episode_chunk=chunk, video_key=key, episode_index=episode))
    return out


def download_source(repo: str, info: dict, episodes: list[int], workers: int) -> None:
    paths = [p for ep in episodes for p in episode_files(info, ep)]
    print(f"  downloading {len(paths):,} files ({len(episodes):,} episodes) with {workers} workers")
    done = [0]
    started = time.monotonic()

    def grab(rel: str) -> None:
        fetch(repo, rel)
        with _print_lock:
            done[0] += 1

    # Heartbeat on a timer, not on a file-count boundary. Count-based progress is
    # indistinguishable from a hang: an earlier run sat silent for 23 minutes between
    # two 500-file marks and looked identical to slow-but-working.
    stop = threading.Event()

    def heartbeat() -> None:
        last = -1
        while not stop.wait(60):
            n = done[0]
            el = time.monotonic() - started
            rate = n / el if el else 0
            eta = (len(paths) - n) / rate if rate > 0 else float("inf")
            state = "STALLED" if n == last else "ok"
            print(f"    {n:,}/{len(paths):,} ({n / len(paths) * 100:.0f}%) "
                  f"{rate:.1f} files/s  eta {eta / 60:.1f} min  "
                  f"[{_retries[0]} retries] {state}", flush=True)
            last = n

    hb = threading.Thread(target=heartbeat, daemon=True)
    hb.start()
    try:
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            errs = [f for f in cf.as_completed(ex.submit(grab, p) for p in paths)
                    if f.exception()]
    finally:
        stop.set()
    if errs:
        raise SystemExit(f"  {len(errs)} downloads failed after retries, "
                         f"first: {errs[0].exception()}")
    el = time.monotonic() - started
    print(f"  downloaded {len(paths):,} files in {el / 60:.1f} min "
          f"({_retries[0]} retries absorbed)")


def assemble(repo: str, info: dict, episodes: list[int], out: pathlib.Path,
             meta_dir: pathlib.Path, extra_meta: dict) -> dict:
    """Copy the selected episodes into `out`, renumbered 0..N-1, with meta rewritten."""
    src_eps = {int(e["episode_index"]): e for e in read_jsonl(meta_dir / "episodes.jsonl")}
    src_stats = {int(s["episode_index"]): s for s in read_jsonl(meta_dir / "episodes_stats.jsonl")}
    video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    chunks = info["chunks_size"]

    out.mkdir(parents=True, exist_ok=True)
    global_index = 0
    for new, orig in enumerate(episodes):
        src_chunk, dst_chunk = orig // chunks, new // chunks
        src_pq = fetch(repo, info["data_path"].format(
            episode_chunk=src_chunk, episode_index=orig))
        dst_pq = out / info["data_path"].format(episode_chunk=dst_chunk, episode_index=new)
        dst_pq.parent.mkdir(parents=True, exist_ok=True)

        frames = pd.read_parquet(src_pq)
        expect = int(src_eps[orig]["length"])
        if len(frames) != expect:
            raise SystemExit(f"episode {orig}: parquet has {len(frames)} rows, meta says {expect}")
        frames["episode_index"] = new
        # `index` is the dataset-global frame counter; it must be contiguous in the copy.
        frames["index"] = global_index + frames["frame_index"].to_numpy()
        global_index += len(frames)
        frames.to_parquet(dst_pq, index=False)

        for key in video_keys:
            src_v = fetch(repo, info["video_path"].format(
                episode_chunk=src_chunk, video_key=key, episode_index=orig))
            dst_v = out / info["video_path"].format(
                episode_chunk=dst_chunk, video_key=key, episode_index=new)
            dst_v.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_v, dst_v)

        if (new + 1) % 500 == 0:
            print(f"    assembled {new + 1:,}/{len(episodes):,}", flush=True)

    total_frames = global_index
    new_info = dict(info)
    new_info["total_episodes"] = len(episodes)
    new_info["total_frames"] = total_frames
    new_info["total_videos"] = len(episodes) * len(video_keys)
    new_info["total_chunks"] = (len(episodes) - 1) // chunks + 1
    new_info["splits"] = {"train": f"0:{len(episodes)}"}
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "meta" / "info.json").write_text(json.dumps(new_info, indent=4))

    write_jsonl(out / "meta" / "episodes.jsonl",
                [{**src_eps[o], "episode_index": n} for n, o in enumerate(episodes)])
    write_jsonl(out / "meta" / "episodes_stats.jsonl",
                [{**src_stats[o], "episode_index": n} for n, o in enumerate(episodes)])
    shutil.copy2(meta_dir / "tasks.jsonl", out / "meta" / "tasks.jsonl")

    (out / "source_episodes.json").write_text(json.dumps({
        "source_repo": repo,
        "episodes": episodes,
        "index_map": {str(o): n for n, o in enumerate(episodes)},
        "total_frames": total_frames,
        "hours": total_frames / FPS / 3600,
        **extra_meta,
        "note": "'episodes' are ORIGINAL indices in the source repo, in output order; "
                "inside this folder they are renumbered 0..N-1.",
    }, indent=2))
    return {"episodes": len(episodes), "frames": total_frames}


def verify(out: pathlib.Path) -> bool:
    info = json.loads((out / "meta" / "info.json").read_text())
    eps = read_jsonl(out / "meta" / "episodes.jsonl")
    stats = read_jsonl(out / "meta" / "episodes_stats.jsonl")
    video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    problems = []

    if [e["episode_index"] for e in eps] != list(range(len(eps))):
        problems.append("episodes.jsonl indices are not contiguous 0..N-1")
    if [s["episode_index"] for s in stats] != list(range(len(stats))):
        problems.append("episodes_stats.jsonl indices are not contiguous 0..N-1")
    if len(eps) != info["total_episodes"]:
        problems.append(f"episodes.jsonl has {len(eps)}, info says {info['total_episodes']}")
    if sum(e["length"] for e in eps) != info["total_frames"]:
        problems.append("episode lengths do not sum to info.total_frames")

    running = 0
    for e in eps:
        n = e["episode_index"]
        chunk = n // info["chunks_size"]
        pq = out / info["data_path"].format(episode_chunk=chunk, episode_index=n)
        if not pq.is_file():
            problems.append(f"missing parquet {pq.name}")
            continue
        df = pd.read_parquet(pq, columns=["episode_index", "index", "frame_index"])
        if len(df) != e["length"]:
            problems.append(f"ep {n}: {len(df)} rows vs meta {e['length']}")
        if (df["episode_index"] != n).any():
            problems.append(f"ep {n}: episode_index column not rewritten")
        if df["index"].iloc[0] != running:
            problems.append(f"ep {n}: global index starts at {df['index'].iloc[0]}, expected {running}")
        running += len(df)
        for key in video_keys:
            v = out / info["video_path"].format(episode_chunk=chunk, video_key=key, episode_index=n)
            if not v.is_file():
                problems.append(f"missing video {key}/{v.name}")

    if running != info["total_frames"]:
        problems.append(f"global index ran to {running}, info says {info['total_frames']}")

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"  {out.name}: {info['total_episodes']:,} eps  {info['total_frames']:,} frames  "
          f"{info['total_frames'] / FPS / 3600:.3f} h  {size / 2**30:.2f} GiB")
    if problems:
        print(f"  FAILED ({len(problems)} problems):")
        for p in problems[:15]:
            print(f"    - {p}")
        if len(problems) > 15:
            print(f"    ... +{len(problems) - 15} more")
        return False
    print("  verified: indices contiguous, all parquet+video present, frame counts match")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default="manifest.json")
    p.add_argument("--out-root", default="/workspace/data/yam7h")
    p.add_argument("--workers", type=int, default=8,
                   help="parallel downloads; >8 tends to trigger Hub 429s")
    p.add_argument("--only", choices=["teleop", "ego"], help="build just one source")
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()

    out_root = pathlib.Path(args.out_root)

    if args.verify_only:
        ok = True
        for name in ("teleop", "ego"):
            d = out_root / name
            if d.is_dir():
                ok &= verify(d)
            else:
                print(f"  {name}: not built")
        sys.exit(0 if ok else 1)

    if os.environ.get("HF_TOKEN"):
        print("auth: using HF_TOKEN (higher Hub rate limit)")
    else:
        print("auth: ANONYMOUS -- low rate limit, expect many 429 retries. "
              "Set HF_TOKEN to speed this up.")

    manifest = json.loads(pathlib.Path(args.manifest).read_text())
    results = {}
    for name, spec in manifest["sources"].items():
        if args.only and name != args.only:
            continue
        repo = spec["repo"]
        episodes = [int(e) for e in spec["episodes"]]
        print(f"\n=== {name}: {repo} ({len(episodes):,} episodes) ===")
        meta_dir = pathlib.Path(snapshot_download(
            repo, repo_type="dataset", allow_patterns=["meta/*"], max_workers=8)) / "meta"
        info = json.loads((meta_dir / "info.json").read_text())

        download_source(repo, info, episodes, args.workers)
        print("  assembling...")
        extra = {k: v for k, v in spec.items() if k not in ("repo", "episodes")}
        extra["seed"] = manifest.get("seed")
        extra["min_eps_per_object"] = manifest.get("min_eps_per_object")
        results[name] = assemble(repo, info, episodes, out_root / name, meta_dir, extra)

    print("\n=== verify ===")
    ok = True
    for name in results:
        ok &= verify(out_root / name)

    total = sum(r["frames"] for r in results.values())
    print(f"\nTOTAL {total:,} frames  {total / FPS / 3600:.3f} h  ->  {out_root}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

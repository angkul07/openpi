#!/usr/bin/env python
"""Carve a teleop holdout out of the mixture so storage lands at a target ratio.

The two datasets are stored at ~33.4% teleop / 66.6% ego by duration. Holding out
teleop episodes for evaluation also brings the *training* storage ratio to 30/70:

    teleop_keep / (teleop_keep + ego) = target        =>  teleop_keep = target/(1-target) * ego

Episodes are selected to hit the target by DURATION, not by episode count -- lengths
vary (teleop episodes run ~11s on average but not uniformly), so dropping 14.4% of the
episodes would not drop 14.4% of the hours.

What this does NOT do: it does not modify or delete anything in the source dataset. The
held-out episodes are *copied* out, and the training run simply never samples them
(`MixtureSource.exclude_episodes`). Rewriting the source in place would mean renumbering
every episode in its parquet files -- precisely the kind of conversion this pipeline
refuses to do -- and would make the split irreversible. The cost is ~0.4 GB of duplicated
video, nothing against 200 GB of disk.

The copy *is* renumbered 0..K-1 so it stands alone as a loadable v2.1 dataset (lerobot
probes `range(total_episodes)` for local files, so a folder numbered 12,20,44,... is
unopenable). Original indices are recorded in holdout_episodes.json.

Usage:
    uv run vast_run/make_holdout.py --dry-run          # numbers only, touches nothing
    uv run vast_run/make_holdout.py                    # write the holdout dataset
"""

import argparse
import json
import os
import pathlib
import shutil

import numpy as np
import pandas as pd


def _lerobot_home() -> pathlib.Path:
    return pathlib.Path(os.environ.get("HF_LEROBOT_HOME", pathlib.Path.home() / ".cache/huggingface/lerobot"))


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def select_holdout(lengths: dict[int, int], target_frames: int, seed: int) -> list[int]:
    """Pick episodes whose total length lands as close to `target_frames` as possible.

    Shuffle deterministically, then take episodes until adding the next one would
    overshoot by more than stopping short would undershoot. Random order (rather than
    longest-first) keeps the holdout an unbiased sample of the teleop distribution --
    a duration-greedy pick would bias it toward long episodes.
    """
    order = list(lengths)
    np.random.default_rng(seed).shuffle(order)

    chosen: list[int] = []
    total = 0
    for ep in order:
        if total >= target_frames:
            break
        if total + lengths[ep] > target_frames and (total + lengths[ep]) - target_frames > target_frames - total:
            continue  # overshoots worse than stopping here; try a shorter episode
        chosen.append(ep)
        total += lengths[ep]
    return sorted(chosen)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teleop-repo", default="angkul07/abc-teleop")
    p.add_argument("--ego-repo", default="angkul07/EgoDex-PickPlace-YAM-14dof-multiview")
    p.add_argument("--target-frac", type=float, default=0.30, help="teleop share of training storage")
    p.add_argument("--out", default="/workspace/abc-teleop-holdout")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    args = p.parse_args()

    home = _lerobot_home()
    teleop_root, ego_root = home / args.teleop_repo, home / args.ego_repo
    teleop_info = json.loads((teleop_root / "meta" / "info.json").read_text())
    ego_info = json.loads((ego_root / "meta" / "info.json").read_text())

    fps = teleop_info["fps"]
    if ego_info["fps"] != fps:
        raise SystemExit(f"fps mismatch: teleop {fps} vs ego {ego_info['fps']}")

    teleop_frames, ego_frames = teleop_info["total_frames"], ego_info["total_frames"]
    keep_target = round(args.target_frac / (1 - args.target_frac) * ego_frames)
    holdout_target = teleop_frames - keep_target
    if holdout_target <= 0:
        raise SystemExit(f"teleop is already below {args.target_frac:.0%}; nothing to hold out")

    episodes = _read_jsonl(teleop_root / "meta" / "episodes.jsonl")
    lengths = {int(e["episode_index"]): int(e["length"]) for e in episodes}
    if sum(lengths.values()) != teleop_frames:
        raise SystemExit(f"episodes.jsonl sums to {sum(lengths.values())}, info.json says {teleop_frames}")

    holdout = select_holdout(lengths, holdout_target, args.seed)
    held_frames = sum(lengths[e] for e in holdout)
    kept_frames = teleop_frames - held_frames
    ratio = kept_frames / (kept_frames + ego_frames)

    hrs = lambda n: n / fps / 3600  # noqa: E731
    print(f"teleop  : {teleop_frames:>9,} frames  {hrs(teleop_frames):6.3f} h  ({len(lengths)} episodes)")
    print(f"ego     : {ego_frames:>9,} frames  {hrs(ego_frames):6.3f} h  ({ego_info['total_episodes']} episodes)")
    print(f"current : teleop is {teleop_frames / (teleop_frames + ego_frames):.2%} of storage")
    print(f"target  : {args.target_frac:.0%} -> keep {keep_target:,} frames, hold out ~{holdout_target:,}")
    print()
    print(
        f"holdout : {len(holdout):>4} episodes  {held_frames:>9,} frames  {hrs(held_frames):6.3f} h "
        f"({held_frames / teleop_frames:.2%} of teleop, {held_frames - holdout_target:+,} vs target)"
    )
    print(f"train   : {len(lengths) - len(holdout):>4} episodes  {kept_frames:>9,} frames  {hrs(kept_frames):6.3f} h")
    print(f"result  : teleop is {ratio:.2%} of training storage (target {args.target_frac:.0%})")
    print()
    print("write into configs/fd/teleop_holdout.json as {\"episodes\": [...]}:")
    print("(" + ", ".join(str(e) for e in holdout) + ")")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    out = pathlib.Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} already exists -- remove it first (refusing to overwrite a holdout)")

    # Write the holdout as a *self-contained* v2.1 dataset with episodes renumbered
    # 0..K-1. Keeping the original indices would look tidier but produces a dataset
    # lerobot cannot open: `get_episodes_file_paths` probes `range(total_episodes)`, so a
    # 249-episode folder numbered 12,20,44,... sends it looking for episode 0, the local
    # file check fails, and it silently falls back to downloading from the hub (404/401).
    # Renumbering keeps the eval side free of special-case loading. The original indices
    # are preserved in holdout_episodes.json -- and remain what `exclude_episodes` uses,
    # since those refer to the source dataset.
    video_keys = [k for k, v in teleop_info["features"].items() if v["dtype"] == "video"]
    chunks_size = teleop_info["chunks_size"]
    index_map = {orig: new for new, orig in enumerate(holdout)}

    global_index = 0
    for new, orig in enumerate(holdout):
        src_chunk, dst_chunk = orig // chunks_size, new // chunks_size
        src_parquet = teleop_root / teleop_info["data_path"].format(episode_chunk=src_chunk, episode_index=orig)
        dst_parquet = out / teleop_info["data_path"].format(episode_chunk=dst_chunk, episode_index=new)
        dst_parquet.parent.mkdir(parents=True, exist_ok=True)

        frames = pd.read_parquet(src_parquet)
        if len(frames) != lengths[orig]:
            raise SystemExit(f"episode {orig}: parquet has {len(frames)} rows, meta says {lengths[orig]}")
        frames["episode_index"] = new
        # `index` is the dataset-global frame counter; it must be contiguous in the copy.
        frames["index"] = global_index + frames["frame_index"].to_numpy()
        global_index += len(frames)
        frames.to_parquet(dst_parquet, index=False)

        for key in video_keys:
            src_v = teleop_root / teleop_info["video_path"].format(
                episode_chunk=src_chunk, video_key=key, episode_index=orig
            )
            dst_v = out / teleop_info["video_path"].format(episode_chunk=dst_chunk, video_key=key, episode_index=new)
            dst_v.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_v, dst_v)

        if (new + 1) % 50 == 0:
            print(f"  wrote {new + 1}/{len(holdout)} episodes")

    if global_index != held_frames:
        raise SystemExit(f"wrote {global_index} frames, expected {held_frames}")

    info = dict(teleop_info)
    info["total_episodes"] = len(holdout)
    info["total_frames"] = held_frames
    info["total_videos"] = len(holdout) * len(video_keys)
    info["total_chunks"] = (len(holdout) - 1) // chunks_size + 1
    info["splits"] = {"train": f"0:{len(holdout)}"}
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    by_index = {int(e["episode_index"]): e for e in episodes}
    _write_jsonl(
        out / "meta" / "episodes.jsonl",
        [{**by_index[orig], "episode_index": new} for new, orig in enumerate(holdout)],
    )
    stats_by_index = {int(s["episode_index"]): s for s in _read_jsonl(teleop_root / "meta" / "episodes_stats.jsonl")}
    _write_jsonl(
        out / "meta" / "episodes_stats.jsonl",
        [{**stats_by_index[orig], "episode_index": new} for new, orig in enumerate(holdout)],
    )
    shutil.copy2(teleop_root / "meta" / "tasks.jsonl", out / "meta" / "tasks.jsonl")

    (out / "holdout_episodes.json").write_text(
        json.dumps(
            {
                "source_repo": args.teleop_repo,
                "seed": args.seed,
                "target_frac": args.target_frac,
                "episodes": holdout,
                "index_map": {str(orig): new for orig, new in index_map.items()},
                "held_frames": held_frames,
                "held_hours": hrs(held_frames),
                "train_frames_remaining": kept_frames,
                "resulting_teleop_storage_frac": ratio,
                "note": (
                    "'episodes' are ORIGINAL indices in the source dataset -- pass these to "
                    "MixtureSource.exclude_episodes. Inside this folder they are renumbered "
                    "0..K-1 in the same order; 'index_map' is original -> local."
                ),
            },
            indent=2,
        )
    )
    print(f"\nwrote {len(holdout)} episodes ({held_frames:,} frames) to {out}")
    print(f"episode list: {out}/holdout_episodes.json")


if __name__ == "__main__":
    main()

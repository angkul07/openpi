"""Build a gap-stratified, frame-budgeted subset of ego_v21 as a standalone v2.1 dataset.

    python build_ego_subset.py --target-frames 9000 --dest .../ego_gap5_v21 \
        [--restrict-to .../ego_gap10_v21/meta/source.json]

WHY PRE-BUILD INSTEAD OF USING `MixtureSource.holdout_fraction`:
`select_holdout_episodes` is a uniform `rng.choice` over episodes, so it would erode the
gap stratification this pool exists to have -- and it would do so on mix10, the arm that
is the clean A/B against sim10. `--restrict-to` also makes the smaller pool a STRICT
SUBSET of the larger one, which holdout_fraction cannot do (two independent draws are
not nested), so mix10's ego is exactly mix20's ego minus some episodes.
"""

import argparse
import collections
import csv
import json
import pathlib
import random
import re
import shutil

import pandas as pd

SRC = pathlib.Path("/workspace/mm/ego_v21")
FILLS = pathlib.Path("/workspace/gap-poc/annotations/gap_fills.csv")
VIDEO_KEYS = ("observation.images.front", "observation.images.wrist")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-frames", type=int, required=True)
    ap.add_argument("--dest", type=pathlib.Path, required=True)
    ap.add_argument("--restrict-to", type=pathlib.Path, default=None,
                    help="source.json of a parent subset; selection is drawn only from its episodes")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dst = a.dest

    info = json.loads((SRC / "meta/info.json").read_text())
    src_eps = {e["episode_index"]: e for e in
               (json.loads(x) for x in (SRC / "meta/episodes.jsonl").read_text().splitlines() if x.strip())}
    src_tasks = {t["task_index"]: t["task"] for t in
                 (json.loads(x) for x in (SRC / "meta/tasks.jsonl").read_text().splitlines() if x.strip())}

    sig = collections.defaultdict(set)
    for r in csv.DictReader(FILLS.open()):
        m = re.search(r"_(\d+)$", r["ego_clip"])
        if not m:
            raise SystemExit(f"clip stem has no trailing index: {r['ego_clip']}")
        sig[int(m.group(1))].add(r["gap"])

    pool = sorted(sig)
    if a.restrict_to:
        parent = set(json.loads(a.restrict_to.read_text())["kept_source_episodes"])
        pool = [e for e in pool if e in parent]
        print(f"restricted to parent subset: {len(pool)} episodes")
    pool_frames = sum(src_eps[e]["length"] for e in pool)
    print(f"pool   : {len(pool)} eps / {pool_frames} frames ({pool_frames / 1800:.2f} min)")
    print(f"target : {a.target_frames} frames ({a.target_frames / 1800:.2f} min)")

    groups = collections.defaultdict(list)
    for e in pool:
        groups[frozenset(sig[e])].append(e)
    ratio = a.target_frames / pool_frames
    rng = random.Random(a.seed)
    keep, leftover = [], []
    for key in sorted(groups, key=lambda k: (-len(groups[k]), sorted(k))):
        eps = sorted(groups[key])
        rng.shuffle(eps)
        quota, acc = ratio * sum(src_eps[e]["length"] for e in eps), 0
        for e in eps:
            if acc + src_eps[e]["length"] <= quota:
                keep.append(e); acc += src_eps[e]["length"]
            else:
                leftover.append(e)
    have = sum(src_eps[e]["length"] for e in keep)
    leftover.sort(key=lambda e: src_eps[e]["length"])
    for e in list(leftover):
        if have >= a.target_frames:
            break
        if have + src_eps[e]["length"] <= a.target_frames:
            keep.append(e); have += src_eps[e]["length"]; leftover.remove(e)
    keep.sort()
    print(f"\nselected {len(keep)} eps / {have} frames ({have / 1800:.2f} min)\n")

    print(f"{'gap':28} {'pool':>6} {'kept':>6} {'%kept':>7}")
    for g in sorted({g for s in sig.values() for g in s}):
        p = [e for e in pool if g in sig[e]]
        k = [e for e in keep if g in sig[e]]
        pct = 100 * len(k) / len(p) if p else 0.0
        print(f"{g:28} {len(p):6d} {len(k):6d} {pct:6.1f}%")
    print(f"{'ALL (episodes)':28} {len(pool):6d} {len(keep):6d} {100 * len(keep) / len(pool):6.1f}%")

    if dst.exists():
        raise SystemExit(f"{dst} exists -- refusing to overwrite")
    for k in VIDEO_KEYS:
        (dst / "videos/chunk-000" / k).mkdir(parents=True)
    (dst / "data/chunk-000").mkdir(parents=True)
    (dst / "meta").mkdir(parents=True)

    used = []
    for old in keep:
        for t in src_eps[old]["tasks"]:
            if t not in used:
                used.append(t)
    t2n = {t: i for i, t in enumerate(used)}

    src_stats = {}
    sp = SRC / "meta/episodes_stats.jsonl"
    if sp.exists():
        for line in sp.read_text().splitlines():
            if line.strip():
                s = json.loads(line); src_stats[s["episode_index"]] = s

    ep_lines, stat_lines, running, total = [], [], 0, 0
    for new, old in enumerate(keep):
        df = pd.read_parquet(SRC / f"data/chunk-000/episode_{old:06d}.parquet")
        df["episode_index"] = new
        df["index"] = range(running, running + len(df))
        running += len(df)
        df["task_index"] = df["task_index"].map(lambda i: t2n[src_tasks[int(i)]])
        df.to_parquet(dst / f"data/chunk-000/episode_{new:06d}.parquet", index=False)
        for k in VIDEO_KEYS:
            shutil.copy2(SRC / f"videos/chunk-000/{k}/episode_{old:06d}.mp4",
                         dst / f"videos/chunk-000/{k}/episode_{new:06d}.mp4")
        e = dict(src_eps[old]); e["episode_index"] = new
        if len(df) != e["length"]:
            raise SystemExit(f"ep {old}: {len(df)} rows vs meta {e['length']}")
        ep_lines.append(json.dumps(e)); total += e["length"]
        if old in src_stats:
            s = dict(src_stats[old]); s["episode_index"] = new
            stat_lines.append(json.dumps(s))

    (dst / "meta/episodes.jsonl").write_text("\n".join(ep_lines) + "\n")
    if stat_lines:
        (dst / "meta/episodes_stats.jsonl").write_text("\n".join(stat_lines) + "\n")
    (dst / "meta/tasks.jsonl").write_text(
        "\n".join(json.dumps({"task_index": i, "task": t}) for i, t in enumerate(used)) + "\n")

    out = dict(info)
    out.update(total_episodes=len(keep), total_frames=total, total_tasks=len(used),
               total_videos=len(keep) * len(VIDEO_KEYS), total_chunks=1,
               splits={"train": f"0:{len(keep)}"})
    (dst / "meta/info.json").write_text(json.dumps(out, indent=4) + "\n")

    prov = json.loads((SRC / "meta/source.json").read_text()) if (SRC / "meta/source.json").exists() else {}
    prov.update(derived_from="ego_v21",
                filter="gap-poc-so101 union of all six gaps, gap-stratified frame-budgeted subset",
                target_frames=a.target_frames, seed=a.seed,
                restricted_to=str(a.restrict_to) if a.restrict_to else None,
                kept_source_episodes=keep)
    (dst / "meta/source.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(f"\nDONE  {len(keep)} eps / {total} frames ({total / 1800:.2f} min), {len(used)} tasks -> {dst}")


if __name__ == "__main__":
    main()

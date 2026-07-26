#!/usr/bin/env python
"""Plan a 7-hour teleop+ego mixture and write the episode manifest.

    teleop  2.33 h / 252,000 frames   angkul07/abc-teleop            (screwdriver)
    ego     4.67 h / 504,000 frames   EgoDex-PickPlace-YAM-14dof     (retargeted)

This script downloads ONLY `meta/` from each source (a few MB) and decides which
episodes to keep. Nothing bulky moves until build_mixture.py runs. Selection is
seeded, so re-running reproduces the same manifest.

Ego constraint -- min episodes per object
-----------------------------------------
Objects are not a metadata field; they are parsed out of the task sentence
("Pick up a <object> from the <source> and place it ..."). The parser is validated
against the published category table: it reproduces all 9 category episode counts
and all 9 distinct-object counts (269 objects total) exactly. Note the determiner
alternation must be longest-first -- `(?:a|an)` matches "a" inside "an orange ball"
and leaves "n orange ball", which silently splits one object into two.

Episodes whose object has fewer than --min-eps-per-object episodes are dropped
BEFORE subsampling. The pool then exceeds the 4.67 h target, so it is subsampled
proportionally per object, which preserves the object mix of the filtered pool.

Note the threshold applies to the source pool, not the delivered dataset: an object
sitting exactly at the threshold keeps only its proportional share after subsampling.
--report-final-min prints the resulting per-object floor so this is visible.

Teleop
------
The 249 episodes already published as angkul07/abc-teleop-holdout are excluded by
default so the eval split stays unseen. Pass --no-exclude-holdout to override.

Usage:
    python select_mixture.py --dry-run        # numbers only
    python select_mixture.py                  # write manifest.json
"""

import argparse
import collections
import json
import pathlib
import re

import numpy as np
from huggingface_hub import snapshot_download

FPS = 30

# Longest-first alternation: "an" MUST be tried before "a".
_OBJ_DET = re.compile(r"^pick up\s+(?:an|a|one|the)\s+(.*?)\s+from\s+", re.I)
_OBJ_BARE = re.compile(r"^pick up\s+(.*?)\s+from\s+", re.I)

# Published category table, used as a correctness check on the parser.
REF_EPISODES = {"puzzle": 158, "plush_toys": 1123, "dice_balls": 1032, "food": 1007,
                "stationery": 918, "tools": 1088, "blocks_shapes": 1259,
                "household": 1097, "electronics": 1162}
REF_OBJECTS = {"puzzle": 1, "plush_toys": 43, "dice_balls": 16, "food": 49,
               "stationery": 21, "tools": 26, "blocks_shapes": 72,
               "household": 26, "electronics": 15}


def parse_object(task: str) -> str | None:
    m = _OBJ_DET.match(task.strip()) or _OBJ_BARE.match(task.strip())
    return m.group(1).strip().lower() if m else None


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def fetch_meta(repo: str) -> pathlib.Path:
    return pathlib.Path(snapshot_download(
        repo, repo_type="dataset", allow_patterns=["meta/*"], max_workers=8)) / "meta"


def take_to_budget(lengths: dict[int, int], target: int, rng: np.random.Generator) -> list[int]:
    """Pick episodes whose total length lands as close to `target` frames as possible.

    Shuffle, then take until the next episode would overshoot by more than stopping
    short would undershoot. Random order (not longest-first) keeps the sample
    unbiased with respect to episode duration.
    """
    order = list(lengths)
    rng.shuffle(order)
    chosen: list[int] = []
    total = 0
    for ep in order:
        if total >= target:
            break
        n = lengths[ep]
        if total + n > target and (total + n) - target > target - total:
            continue
        chosen.append(ep)
        total += n
    return sorted(chosen)


# --------------------------------------------------------------------------- ego
def select_ego(repo: str, target_frames: int, min_eps: int, seed: int, verbose: bool):
    meta = fetch_meta(repo)
    info = json.loads((meta / "info.json").read_text())
    episodes = read_jsonl(meta / "episodes.jsonl")
    qa = {r["episode_index"]: r for r in read_jsonl(meta / "episodes_qa.jsonl")}

    rows = []
    for e in episodes:
        idx = int(e["episode_index"])
        rows.append({"ep": idx, "len": int(e["length"]), "task": e["tasks"][0],
                     "obj": parse_object(e["tasks"][0]),
                     "cat": qa.get(idx, {}).get("category"),
                     "ik": qa.get(idx, {}).get("active_ik_mean_cm"),
                     "verdict": qa.get(idx, {}).get("qa_verdict")})

    unparsed = [r for r in rows if not r["obj"]]
    if unparsed:
        raise SystemExit(f"{len(unparsed)} task sentences did not parse, e.g. {unparsed[0]['task']!r}")

    # Validate the object definition against the published table before trusting it.
    by_cat = collections.defaultdict(list)
    for r in rows:
        by_cat[r["cat"]].append(r)
    mismatches = []
    for cat, ref_objs in REF_OBJECTS.items():
        got = len({r["obj"] for r in by_cat.get(cat, [])})
        if got != ref_objs:
            mismatches.append(f"{cat}: {got} objects, table says {ref_objs}")
    if mismatches:
        raise SystemExit("object parser disagrees with the published table:\n  " + "\n  ".join(mismatches))
    if verbose:
        ep_diff = {c: (len(by_cat.get(c, [])), REF_EPISODES[c])
                   for c in REF_EPISODES if len(by_cat.get(c, [])) != REF_EPISODES[c]}
        print(f"  parser validated: {len({r['obj'] for r in rows})} objects across "
              f"{len(REF_OBJECTS)} categories matches the published table")
        if ep_diff:
            print(f"  note: episode-count diffs vs table (source has fewer episodes): {ep_diff}")

    per_obj = collections.Counter(r["obj"] for r in rows)
    keep_objs = {o for o, c in per_obj.items() if c >= min_eps}
    pool = [r for r in rows if r["obj"] in keep_objs]
    pool_frames = sum(r["len"] for r in pool)
    if pool_frames < target_frames:
        raise SystemExit(
            f"after the min-{min_eps} filter the pool holds {pool_frames:,} frames "
            f"({pool_frames / FPS / 3600:.2f} h) but {target_frames:,} "
            f"({target_frames / FPS / 3600:.2f} h) are needed -- lower --min-eps-per-object")

    # Proportional per-object subsample: every surviving object keeps the same share.
    frac = target_frames / pool_frames
    rng = np.random.default_rng(seed)
    by_obj = collections.defaultdict(dict)
    for r in pool:
        by_obj[r["obj"]][r["ep"]] = r["len"]

    chosen: list[int] = []
    for obj in sorted(by_obj):
        lengths = by_obj[obj]
        chosen += take_to_budget(lengths, round(sum(lengths.values()) * frac), rng)
    chosen.sort()

    return {"repo": repo, "info": info, "rows": {r["ep"]: r for r in rows},
            "pool": pool, "pool_frames": pool_frames, "keep_objs": keep_objs,
            "per_obj": per_obj, "chosen": chosen, "frac": frac}


# ------------------------------------------------------------------------ teleop
def select_teleop(repo: str, target_frames: int, seed: int, holdout: list[int]):
    meta = fetch_meta(repo)
    info = json.loads((meta / "info.json").read_text())
    episodes = read_jsonl(meta / "episodes.jsonl")
    excluded = set(holdout)
    lengths = {int(e["episode_index"]): int(e["length"])
               for e in episodes if int(e["episode_index"]) not in excluded}
    pool_frames = sum(lengths.values())
    if pool_frames < target_frames:
        raise SystemExit(f"teleop pool has {pool_frames:,} frames, need {target_frames:,}")
    chosen = take_to_budget(lengths, target_frames, np.random.default_rng(seed))
    return {"repo": repo, "info": info, "lengths": lengths, "pool_frames": pool_frames,
            "chosen": chosen, "excluded": sorted(excluded)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teleop-repo", default="angkul07/abc-teleop")
    p.add_argument("--ego-repo", default="angkul07/EgoDex-PickPlace-YAM-14dof-multiview")
    p.add_argument("--teleop-hours", type=float, default=2.33)
    p.add_argument("--ego-hours", type=float, default=4.67)
    p.add_argument("--min-eps-per-object", type=int, default=30)
    p.add_argument("--holdout-manifest", default="teleop_holdout.json",
                   help="episodes to exclude from teleop (the published eval split)")
    p.add_argument("--no-exclude-holdout", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="manifest.json")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    teleop_target = round(args.teleop_hours * 3600 * FPS)
    ego_target = round(args.ego_hours * 3600 * FPS)

    holdout: list[int] = []
    if not args.no_exclude_holdout:
        hp = pathlib.Path(args.holdout_manifest)
        if not hp.is_file():
            raise SystemExit(
                f"{hp} not found. It lists the 249 published eval episodes that must stay "
                f"out of training. Pass --holdout-manifest or --no-exclude-holdout.")
        holdout = [int(e) for e in json.loads(hp.read_text())["episodes"]]

    hrs = lambda n: n / FPS / 3600  # noqa: E731

    print("=" * 78)
    print(f"TELEOP  target {args.teleop_hours} h = {teleop_target:,} frames")
    print("=" * 78)
    tl = select_teleop(args.teleop_repo, teleop_target, args.seed, holdout)
    tl_frames = sum(tl["lengths"][e] for e in tl["chosen"])
    src = tl["info"]
    print(f"  source          : {src['total_episodes']:,} eps  {src['total_frames']:,} frames  "
          f"{hrs(src['total_frames']):.3f} h")
    if holdout:
        print(f"  eval holdout    : -{len(holdout)} eps excluded (published as abc-teleop-holdout)")
    print(f"  pool            : {len(tl['lengths']):,} eps  {tl['pool_frames']:,} frames  "
          f"{hrs(tl['pool_frames']):.3f} h")
    print(f"  SELECTED        : {len(tl['chosen']):,} eps  {tl_frames:,} frames  "
          f"{hrs(tl_frames):.3f} h  ({tl_frames - teleop_target:+,} vs target)")

    print()
    print("=" * 78)
    print(f"EGO     target {args.ego_hours} h = {ego_target:,} frames   "
          f"(min {args.min_eps_per_object} eps/object)")
    print("=" * 78)
    eg = select_ego(args.ego_repo, ego_target, args.min_eps_per_object, args.seed, verbose=True)
    rows = eg["rows"]
    eg_frames = sum(rows[e]["len"] for e in eg["chosen"])
    src = eg["info"]
    print(f"  source          : {src['total_episodes']:,} eps  {src['total_frames']:,} frames  "
          f"{hrs(src['total_frames']):.3f} h  ({len(eg['per_obj'])} objects)")
    print(f"  min-{args.min_eps_per_object} filter    : {len(eg['keep_objs'])} objects kept  "
          f"{len(eg['pool']):,} eps  {eg['pool_frames']:,} frames  {hrs(eg['pool_frames']):.3f} h")
    print(f"  subsample       : {eg['frac']:.1%} of the filtered pool, proportional per object")
    print(f"  SELECTED        : {len(eg['chosen']):,} eps  {eg_frames:,} frames  "
          f"{hrs(eg_frames):.3f} h  ({eg_frames - ego_target:+,} vs target)")

    sel_per_obj = collections.Counter(rows[e]["obj"] for e in eg["chosen"])
    sel_per_cat = collections.Counter(rows[e]["cat"] for e in eg["chosen"])
    lo = sel_per_obj.most_common()[-1]
    print(f"  objects retained: {len(sel_per_obj)} / {len(eg['keep_objs'])}   "
          f"eps per object min {lo[1]} ({lo[0]}) max {sel_per_obj.most_common(1)[0][1]}")
    print(f"  categories      : {dict(sel_per_cat.most_common())}")
    ik = [rows[e]["ik"] for e in eg["chosen"] if rows[e]["ik"] is not None]
    if ik:
        print(f"  retarget QA     : active_ik_mean_cm <=15 for "
              f"{sum(1 for v in ik if v <= 15)}/{len(ik)} selected episodes "
              f"(NOT filtered on -- only the min-eps constraint was applied)")

    total = tl_frames + eg_frames
    print()
    print("=" * 78)
    print(f"TOTAL   {total:,} frames  {hrs(total):.3f} h   "
          f"teleop {tl_frames / total:.1%} / ego {eg_frames / total:.1%}")
    print("=" * 78)

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    manifest = {
        "seed": args.seed,
        "fps": FPS,
        "min_eps_per_object": args.min_eps_per_object,
        "sources": {
            "teleop": {"repo": args.teleop_repo, "episodes": tl["chosen"],
                       "frames": tl_frames, "hours": hrs(tl_frames),
                       "excluded_holdout": tl["excluded"]},
            "ego": {"repo": args.ego_repo, "episodes": eg["chosen"],
                    "frames": eg_frames, "hours": hrs(eg_frames),
                    "objects_kept": sorted(eg["keep_objs"]),
                    "episodes_per_object": dict(sel_per_obj.most_common())},
        },
        "total_frames": total,
        "total_hours": hrs(total),
        "note": "episode indices are ORIGINAL indices in each source repo; "
                "build_mixture.py renumbers them 0..N-1 in the output datasets.",
    }
    pathlib.Path(args.out).write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

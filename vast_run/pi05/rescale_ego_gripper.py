"""Rescale the ego gripper channels onto the teleop gripper range, in place.

WHY. Grippers stay absolute through DeltaActions(6,-1,6,-1) and are then quantile-
normalized over the MIXTURE, so both sources' gripper values land on one shared
[-1,1] axis. Measured on the 7h mixture, they do not agree about what "open" means:

    R_grip   teleop open 0.992 -> +0.984 norm | ego open 0.443 -> -0.114 norm
    L_grip   teleop open 0.993 -> +0.986 norm | ego open 0.392 -> -0.216 norm

Ego's "open" sits closer to teleop's CLOSED than to teleop's open. Under pi0.5 that
is materially worse than under pi0-FAST: flow matching with few denoising steps
mode-averages between widely separated modes, and the midpoint of a bimodal
open-gripper target is a half-open hand that grasps nothing. pi0-FAST at least
sampled one discrete token, i.e. one mode.

Ego's absolute gripper scale is a retargeting artifact (it comes from human hand
tracking); teleop's is the real YAM gripper. So teleop is the target range.

ANCHORS. Do NOT map mode-to-mode (ego closed/open medians onto teleop's). Teleop's
gripper is near-binary (p50 = 0.986, 13% of frames closed) while ego's is a
continuous aperture spanning [0.02, 0.86]. Mode-to-mode implies gain ~5.0 on
R_grip actions, which saturates every ego value above 0.413 -- about half of all
ego frames -- at the top of the range. This script therefore anchors on robust
percentiles (p1/p99 by default), which is monotone, preserves ego's within-source
aperture ordering, and clips ~2% instead of ~50%.

STATE AND ACTION ARE MAPPED SEPARATELY. teleop's closed gripper reads 0.270 in
`observation.state` but 0.054 in `action` -- that gap is the gripper servo lag, and
it is real signal. norm_stats carries independent entries for "state" and
"actions", so each ego channel is mapped onto the corresponding teleop channel.
This does break ego's action[t] == state[t+1] identity on dims 6/13 only; nothing
in the training path relies on it (the delta mask leaves grippers absolute, and
only preflight_checks.py's source-classification report reads that identity).

REVERSIBLE. The original gripper columns are dumped to a .npz sidecar before any
write, and --revert restores them. A sentinel file blocks double-application.

Usage (from the openpi repo root):
    uv run vast_run/pi05/rescale_ego_gripper.py                      # dry run, prints the plan
    uv run vast_run/pi05/rescale_ego_gripper.py --apply
    uv run vast_run/pi05/rescale_ego_gripper.py --revert

After --apply the norm stats are stale and MUST be recomputed:
    rm -rf assets/pi05_yam7h_ea/yam7h_p50
    uv run scripts/compute_norm_stats.py --config-name pi05_yam7h_ea \
        --max-frames 200000 --skip-videos
"""

import argparse
import glob
import json
import os
import pathlib
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

GRIPPER_DIMS = (6, 13)
DIM_NAMES = {6: "R_grip", 13: "L_grip"}
COLUMNS = ("observation.state", "action")

DEFAULT_TELEOP = os.environ.get("YAM7H_TELEOP_ROOT", "/workspace/data/yam7h/teleop")
DEFAULT_EGO = os.environ.get("YAM7H_EGO_ROOT", "/workspace/data/yam7h/ego")

SENTINEL = "meta/gripper_rescale.json"
BACKUP = "meta/gripper_rescale_backup.npz"


def parquet_files(root: pathlib.Path) -> list[str]:
    files = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"no parquet under {root}/data")
    return files


def pool(root: pathlib.Path, n: int) -> dict[str, np.ndarray]:
    """Pool a spread of episodes into {column: (N,14)}."""
    files = parquet_files(root)
    step = max(1, len(files) // n)
    acc: dict[str, list] = {c: [] for c in COLUMNS}
    for path in files[::step][:n]:
        t = pq.read_table(path, columns=list(COLUMNS))
        for c in COLUMNS:
            acc[c].append(np.stack(t[c].to_numpy(zero_copy_only=False)).astype(np.float64))
    return {c: np.concatenate(v) for c, v in acc.items()}


def fit(teleop: dict, ego: dict, lo_q: float, hi_q: float) -> dict:
    """Per (column, dim) affine a*x+b mapping ego's [lo_q,hi_q] onto teleop's, plus clip bounds."""
    plan = {}
    for col in COLUMNS:
        for dim in GRIPPER_DIMS:
            t, e = teleop[col][:, dim], ego[col][:, dim]
            t_lo, t_hi = np.percentile(t, [lo_q, hi_q])
            e_lo, e_hi = np.percentile(e, [lo_q, hi_q])
            if e_hi - e_lo < 1e-9:
                raise SystemExit(f"{col} dim {dim}: ego range is degenerate, refusing to fit")
            a = (t_hi - t_lo) / (e_hi - e_lo)
            b = t_lo - a * e_lo
            mapped = a * e + b
            clip_lo, clip_hi = float(t.min()), float(t.max())
            plan[f"{col}:{dim}"] = {
                "column": col,
                "dim": dim,
                "name": DIM_NAMES[dim],
                "a": float(a),
                "b": float(b),
                "clip_lo": clip_lo,
                "clip_hi": clip_hi,
                "teleop_anchor": [float(t_lo), float(t_hi)],
                "ego_anchor": [float(e_lo), float(e_hi)],
                "frac_clipped": float(np.mean((mapped < clip_lo) | (mapped > clip_hi))),
                "ego_closed_before": float(np.median(e[e <= np.median(e)])),
                "ego_open_before": float(np.median(e[e > np.median(e)])),
            }
            m = np.clip(mapped, clip_lo, clip_hi)
            plan[f"{col}:{dim}"]["ego_closed_after"] = float(np.median(m[m <= np.median(m)]))
            plan[f"{col}:{dim}"]["ego_open_after"] = float(np.median(m[m > np.median(m)]))
            plan[f"{col}:{dim}"]["teleop_closed"] = float(np.median(t[t <= np.median(t)]))
            plan[f"{col}:{dim}"]["teleop_open"] = float(np.median(t[t > np.median(t)]))
    return plan


def print_plan(plan: dict, lo_q: float, hi_q: float) -> None:
    print(f"\n=== MAPPING (anchors: ego p{lo_q:g}/p{hi_q:g} -> teleop p{lo_q:g}/p{hi_q:g}) ===")
    for key in sorted(plan):
        p = plan[key]
        print(f"  {p['column']:18s} {p['name']:7s}  x -> {p['a']:.4f}*x {p['b']:+.4f}   "
              f"clip[{p['clip_lo']:.3f},{p['clip_hi']:.3f}]  clipped={p['frac_clipped']:.2%}")
        print(f"      ego closed {p['ego_closed_before']:.3f} -> {p['ego_closed_after']:.3f}  "
              f"(teleop {p['teleop_closed']:.3f})")
        print(f"      ego open   {p['ego_open_before']:.3f} -> {p['ego_open_after']:.3f}  "
              f"(teleop {p['teleop_open']:.3f})")


def rebuild_column(table: pa.Table, name: str, arr: np.ndarray) -> pa.Table:
    """Replace `name` with arr (T,14), preserving the original arrow type."""
    idx = table.schema.get_field_index(name)
    field = table.schema.field(idx)
    flat = pa.array(arr.astype(np.float32).ravel(), type=pa.float32())
    if pa.types.is_fixed_size_list(field.type):
        col = pa.FixedSizeListArray.from_arrays(flat, field.type.list_size)
    else:
        offsets = pa.array(np.arange(arr.shape[0] + 1, dtype=np.int32) * arr.shape[1], type=pa.int32())
        col = pa.ListArray.from_arrays(offsets, flat)
    return table.set_column(idx, field, col)


def apply(ego_root: pathlib.Path, plan: dict, lo_q: float, hi_q: float) -> None:
    files = parquet_files(ego_root)
    backup: dict[str, np.ndarray] = {}
    print(f"\nrewriting {len(files)} parquet files under {ego_root}/data ...")
    for i, path in enumerate(files):
        table = pq.read_table(path)
        rel = str(pathlib.Path(path).relative_to(ego_root))
        changed = False
        for col in COLUMNS:
            arr = np.stack(table[col].to_numpy(zero_copy_only=False)).astype(np.float64)
            backup[f"{rel}|{col}"] = arr[:, list(GRIPPER_DIMS)].astype(np.float32)
            for dim in GRIPPER_DIMS:
                p = plan[f"{col}:{dim}"]
                arr[:, dim] = np.clip(p["a"] * arr[:, dim] + p["b"], p["clip_lo"], p["clip_hi"])
            table = rebuild_column(table, col, arr)
            changed = True
        if changed:
            tmp = path + ".tmp"
            pq.write_table(table, tmp)
            os.replace(tmp, path)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(files)}")

    # Store as one concatenated array + an explicit key/offset index. Passing the dict
    # as **kwargs would unpack arbitrary path-derived strings into savez_compressed's
    # keyword namespace, where a key could collide with `allow_pickle`.
    keys = sorted(backup)
    lengths = [backup[k].shape[0] for k in keys]
    np.savez_compressed(
        ego_root / BACKUP,
        data=np.concatenate([backup[k] for k in keys]).astype(np.float32),
        offsets=np.cumsum([0, *lengths]).astype(np.int64),
        keys=np.array(keys, dtype=object),
    )
    print(f"backup -> {ego_root / BACKUP} ({(ego_root / BACKUP).stat().st_size / 1e6:.1f} MB)")

    update_episode_stats(ego_root)

    (ego_root / SENTINEL).write_text(json.dumps({
        "applied": True, "anchors": [lo_q, hi_q], "gripper_dims": list(GRIPPER_DIMS),
        "plan": plan, "backup": BACKUP,
    }, indent=2))
    print(f"sentinel -> {ego_root / SENTINEL}")


def update_episode_stats(ego_root: pathlib.Path) -> None:
    """Recompute min/max/mean/std for the gripper dims from the rewritten parquet."""
    stats_path = ego_root / "meta" / "episodes_stats.jsonl"
    if not stats_path.is_file():
        print("no episodes_stats.jsonl -- skipping stat refresh")
        return
    by_ep: dict[int, dict] = {}
    for path in parquet_files(ego_root):
        t = pq.read_table(path, columns=[*COLUMNS, "episode_index"])
        ep = int(np.asarray(t["episode_index"].to_numpy(zero_copy_only=False))[0])
        by_ep[ep] = {c: np.stack(t[c].to_numpy(zero_copy_only=False)).astype(np.float64) for c in COLUMNS}

    out = []
    n_patched = 0
    with stats_path.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ep = rec["episode_index"]
            if ep in by_ep:
                for col in COLUMNS:
                    if col not in rec["stats"]:
                        continue
                    arr = by_ep[ep][col]
                    for dim in GRIPPER_DIMS:
                        v = arr[:, dim]
                        rec["stats"][col]["min"][dim] = float(v.min())
                        rec["stats"][col]["max"][dim] = float(v.max())
                        rec["stats"][col]["mean"][dim] = float(v.mean())
                        rec["stats"][col]["std"][dim] = float(v.std())
                n_patched += 1
            out.append(json.dumps(rec))
    shutil.copy2(stats_path, str(stats_path) + ".bak")
    stats_path.write_text("\n".join(out) + "\n")
    print(f"episodes_stats.jsonl: patched {n_patched} episodes (original -> .bak)")


def revert(ego_root: pathlib.Path) -> None:
    sentinel = ego_root / SENTINEL
    backup_path = ego_root / BACKUP
    if not sentinel.is_file() or not backup_path.is_file():
        raise SystemExit(f"nothing to revert: need {sentinel} and {backup_path}")
    npz = np.load(backup_path, allow_pickle=True)
    keys = list(npz["keys"])
    offsets, data = npz["offsets"], npz["data"]
    index = {str(k): (int(offsets[i]), int(offsets[i + 1])) for i, k in enumerate(keys)}
    files = parquet_files(ego_root)
    print(f"reverting {len(files)} files ...")
    for path in files:
        rel = str(pathlib.Path(path).relative_to(ego_root))
        table = pq.read_table(path)
        for col in COLUMNS:
            key = f"{rel}|{col}"
            if key not in index:
                raise SystemExit(f"backup missing {key} -- refusing to partially revert")
            lo, hi = index[key]
            arr = np.stack(table[col].to_numpy(zero_copy_only=False)).astype(np.float64)
            arr[:, list(GRIPPER_DIMS)] = data[lo:hi].astype(np.float64)
            table = rebuild_column(table, col, arr)
        tmp = path + ".tmp"
        pq.write_table(table, tmp)
        os.replace(tmp, path)
    stats_bak = ego_root / "meta" / "episodes_stats.jsonl.bak"
    if stats_bak.is_file():
        shutil.move(str(stats_bak), str(ego_root / "meta" / "episodes_stats.jsonl"))
        print("restored episodes_stats.jsonl")
    sentinel.unlink()
    print("reverted. Recompute norm stats before training.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teleop-root", default=DEFAULT_TELEOP)
    ap.add_argument("--ego-root", default=DEFAULT_EGO)
    ap.add_argument("--episodes", type=int, default=150, help="episodes pooled per source to fit the map")
    ap.add_argument("--lo-q", type=float, default=1.0, help="lower anchor percentile")
    ap.add_argument("--hi-q", type=float, default=99.0, help="upper anchor percentile")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-apply even if the sentinel exists")
    args = ap.parse_args()

    ego_root = pathlib.Path(args.ego_root)
    teleop_root = pathlib.Path(args.teleop_root)

    if args.revert:
        revert(ego_root)
        return

    sentinel = ego_root / SENTINEL
    if sentinel.is_file() and not args.force:
        raise SystemExit(
            f"ALREADY APPLIED: {sentinel} exists. Re-applying would compound the map.\n"
            "Use --revert first, or --force if you really mean it."
        )

    print(f"teleop : {teleop_root}\nego    : {ego_root}\npooling {args.episodes} episodes per source ...")
    teleop = pool(teleop_root, args.episodes)
    ego = pool(ego_root, args.episodes)
    plan = fit(teleop, ego, args.lo_q, args.hi_q)
    print_plan(plan, args.lo_q, args.hi_q)

    if not args.apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply to commit.")
        return
    apply(ego_root, plan, args.lo_q, args.hi_q)
    print("\nDONE. Norm stats are now STALE -- recompute before training:")
    print("  rm -rf assets/pi05_yam7h_ea/yam7h_p50")
    print("  uv run scripts/compute_norm_stats.py --config-name pi05_yam7h_ea --max-frames 200000 --skip-videos")


if __name__ == "__main__":
    main()

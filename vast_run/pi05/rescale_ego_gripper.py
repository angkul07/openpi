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

TWO LAYOUTS ARE SUPPORTED.

  * TWO ROOTS (--teleop-root/--ego-root, the yam7h arms): ego and teleop are separate
    LeRobot datasets, and every parquet under the ego root is rewritten.

  * ONE MERGED ROOT (--merged-root, the pi05_50run_ea dataset): ego and teleop live in
    the SAME root and are told apart by `meta/provenance.jsonl`, which build_50run.py
    writes as {"episode_index", "source", ...}. Only the ego episodes are pooled as the
    source distribution and only they are rewritten; the teleop episodes are the target
    range and are never touched. The sentinel and backup still land in the merged root's
    meta/, and they record which episodes were rewritten so --revert cannot drift.

Usage (from the openpi repo root):
    uv run vast_run/pi05/rescale_ego_gripper.py                      # dry run, prints the plan
    uv run vast_run/pi05/rescale_ego_gripper.py --apply
    uv run vast_run/pi05/rescale_ego_gripper.py --revert

    # merged single-root dataset (pi05_50run_ea):
    uv run vast_run/pi05/rescale_ego_gripper.py --merged-root /workspace/50_run
    uv run vast_run/pi05/rescale_ego_gripper.py --merged-root /workspace/50_run --apply

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


def episode_of(path: str) -> int:
    """episode_000123.parquet -> 123."""
    stem = pathlib.Path(path).stem
    return int(stem.rsplit("_", 1)[1])


def split_merged(root: pathlib.Path, ego_label: str, teleop_label: str) -> tuple[list[str], list[str]]:
    """Split one merged root's parquet into (ego_files, teleop_files) via provenance.jsonl.

    build_50run.py lays ego and teleop out as two contiguous episode blocks in a single
    dataset, so the only record of which is which is meta/provenance.jsonl. Without it we
    would be guessing, and rewriting a teleop episode as if it were ego is unrecoverable
    short of --revert, so a missing/incomplete provenance is a hard error.
    """
    prov = root / "meta" / "provenance.jsonl"
    if not prov.is_file():
        raise SystemExit(f"--merged-root needs {prov} to tell ego from teleop; not found")
    source_of: dict[int, str] = {}
    with prov.open() as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            source_of[int(rec["episode_index"])] = rec["source"]

    labels = set(source_of.values())
    for want in (ego_label, teleop_label):
        if want not in labels:
            raise SystemExit(f"provenance has sources {sorted(labels)}, no '{want}'")

    ego, teleop = [], []
    for path in parquet_files(root):
        ep = episode_of(path)
        if ep not in source_of:
            raise SystemExit(f"episode {ep} ({path}) is absent from {prov}")
        if source_of[ep] == ego_label:
            ego.append(path)
        elif source_of[ep] == teleop_label:
            teleop.append(path)
    if not ego or not teleop:
        raise SystemExit(f"split produced ego={len(ego)} teleop={len(teleop)}; need both")
    print(f"merged root split via provenance: ego={len(ego)} episodes, teleop={len(teleop)} episodes")
    return ego, teleop


def pool(files: list[str], n: int) -> dict[str, np.ndarray]:
    """Pool a spread of episodes into {column: (N,14)}."""
    step = max(1, len(files) // n)
    acc: dict[str, list] = {c: [] for c in COLUMNS}
    for path in files[::step][:n]:
        t = pq.read_table(path, columns=list(COLUMNS))
        for c in COLUMNS:
            acc[c].append(np.stack(t[c].to_numpy(zero_copy_only=False)).astype(np.float64))
    return {c: np.concatenate(v) for c, v in acc.items()}


def modes(x: np.ndarray) -> tuple[float, float]:
    """Closed/open mode medians, split at the MIDRANGE.

    Not at the median: teleop's gripper is ~88% open, so a median split lands inside
    the open mode and reports "closed" as ~0.98. mixture_diagnostics.py splits at
    0.5*(p1+p99) and this must agree with it, or the before/after numbers are lies.
    """
    lo, hi = np.percentile(x, [1, 99])
    mid = 0.5 * (lo + hi)
    low, high = x[x <= mid], x[x > mid]
    return (
        float(np.median(low)) if low.size else float(lo),
        float(np.median(high)) if high.size else float(hi),
    )


def fit(teleop: dict, ego: dict, lo_q: float, hi_q: float) -> dict:
    """Per (column, dim) monotone piecewise-linear map, ego -> teleop.

    Knots are [min, closed_mode, open_mode, max]. An AFFINE map cannot do this job:
    ego's gripper is a continuous aperture (modes ~0.22/0.41 inside [0.02, 0.86])
    while teleop's is near-binary (~0.05/0.99). Anchoring an affine on p1/p99 aligns
    the ranges but leaves the modes ~0.8 apart -- still mode-averaging territory --
    and anchoring it on the modes implies gain ~5 and saturates over half the frames.
    Piecewise-linear pins BOTH modes AND both extremes, so it closes the gap exactly,
    stays monotone (aperture ordering within ego is preserved), needs no clipping,
    and keeps grasp transitions continuous instead of binarizing them.
    """
    plan = {}
    for col in COLUMNS:
        for dim in GRIPPER_DIMS:
            t, e = teleop[col][:, dim], ego[col][:, dim]
            # Robust extremes so a single outlier frame cannot set a knot.
            t_min, t_max = np.percentile(t, [lo_q * 0.1, 100 - (100 - hi_q) * 0.1])
            e_min, e_max = np.percentile(e, [lo_q * 0.1, 100 - (100 - hi_q) * 0.1])
            t_closed, t_open = modes(t)
            e_closed, e_open = modes(e)

            xk = [float(e_min), float(e_closed), float(e_open), float(e_max)]
            yk = [float(t_min), float(t_closed), float(t_open), float(t_max)]
            if not all(xk[i] < xk[i + 1] for i in range(3)):
                raise SystemExit(f"{col} dim {dim}: ego knots not strictly increasing: {xk}")
            if not all(yk[i] < yk[i + 1] for i in range(3)):
                raise SystemExit(f"{col} dim {dim}: teleop knots not strictly increasing: {yk}")

            mapped = np.interp(e, xk, yk)
            m_closed, m_open = modes(mapped)
            plan[f"{col}:{dim}"] = {
                "column": col, "dim": dim, "name": DIM_NAMES[dim],
                "x_knots": xk, "y_knots": yk,
                "ego_closed_before": e_closed, "ego_open_before": e_open,
                "ego_closed_after": m_closed, "ego_open_after": m_open,
                "teleop_closed": t_closed, "teleop_open": t_open,
                # np.interp clamps outside the knot range, so this is the saturated share.
                "frac_saturated": float(np.mean((e < xk[0]) | (e > xk[-1]))),
            }
    return plan


def print_plan(plan: dict, lo_q: float, hi_q: float) -> None:
    print(f"\n=== MAPPING (monotone piecewise-linear; extremes at p{lo_q * 0.1:g}/"
          f"p{100 - (100 - hi_q) * 0.1:g}, plus both modes) ===")
    for key in sorted(plan):
        p = plan[key]
        xs = ", ".join(f"{v:.3f}" for v in p["x_knots"])
        ys = ", ".join(f"{v:.3f}" for v in p["y_knots"])
        print(f"  {p['column']:18s} {p['name']:7s}  saturated={p['frac_saturated']:.2%}")
        print(f"      ego  [{xs}]")
        print(f"      ->   [{ys}]")
        print(f"      closed {p['ego_closed_before']:.3f} -> {p['ego_closed_after']:.3f} "
              f"(teleop {p['teleop_closed']:.3f})   "
              f"open {p['ego_open_before']:.3f} -> {p['ego_open_after']:.3f} "
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


def apply(ego_root: pathlib.Path, files: list[str], plan: dict, lo_q: float, hi_q: float) -> None:
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
                # np.interp clamps outside the knot range, which is the intended
                # saturation at the physical gripper limits.
                arr[:, dim] = np.interp(arr[:, dim], p["x_knots"], p["y_knots"])
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

    update_episode_stats(ego_root, files)

    (ego_root / SENTINEL).write_text(json.dumps({
        "applied": True, "anchors": [lo_q, hi_q], "gripper_dims": list(GRIPPER_DIMS),
        "plan": plan, "backup": BACKUP,
        # Which episodes were rewritten. On a merged root this is the ONLY record that
        # the teleop half was left alone, so --revert must read it back rather than
        # re-deriving the split from provenance.
        "rewritten_episodes": sorted(episode_of(p) for p in files),
    }, indent=2))
    print(f"sentinel -> {ego_root / SENTINEL}")


def update_episode_stats(ego_root: pathlib.Path, files: list[str]) -> None:
    """Recompute min/max/mean/std for the gripper dims from the rewritten parquet."""
    stats_path = ego_root / "meta" / "episodes_stats.jsonl"
    if not stats_path.is_file():
        print("no episodes_stats.jsonl -- skipping stat refresh")
        return
    by_ep: dict[int, dict] = {}
    for path in files:
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
    # Revert exactly what apply() rewrote. Re-globbing the root would, on a merged
    # dataset, also sweep in the teleop half -- which has no backup entry and would
    # abort the revert halfway through, leaving the dataset in a mixed state.
    rewritten = set(json.loads(sentinel.read_text()).get("rewritten_episodes", []))
    files = [p for p in parquet_files(ego_root) if not rewritten or episode_of(p) in rewritten]
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
    ap.add_argument("--merged-root", default=None,
                    help="single root holding BOTH sources; split by meta/provenance.jsonl "
                         "(overrides --ego-root/--teleop-root)")
    ap.add_argument("--ego-label", default="ego", help="provenance 'source' value for ego")
    ap.add_argument("--teleop-label", default="teleop", help="provenance 'source' value for teleop")
    ap.add_argument("--episodes", type=int, default=150, help="episodes pooled per source to fit the map")
    ap.add_argument("--lo-q", type=float, default=1.0, help="lower anchor percentile")
    ap.add_argument("--hi-q", type=float, default=99.0, help="upper anchor percentile")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-apply even if the sentinel exists")
    args = ap.parse_args()

    if args.merged_root:
        # One dataset, two sources: everything (sentinel, backup, stats) hangs off this
        # root, and the ego/teleop distinction comes from provenance rather than layout.
        ego_root = teleop_root = pathlib.Path(args.merged_root)
    else:
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

    if args.merged_root:
        print(f"merged : {ego_root}")
        ego_files, teleop_files = split_merged(ego_root, args.ego_label, args.teleop_label)
    else:
        print(f"teleop : {teleop_root}\nego    : {ego_root}")
        ego_files, teleop_files = parquet_files(ego_root), parquet_files(teleop_root)

    print(f"pooling {args.episodes} episodes per source ...")
    teleop = pool(teleop_files, args.episodes)
    ego = pool(ego_files, args.episodes)
    plan = fit(teleop, ego, args.lo_q, args.hi_q)
    print_plan(plan, args.lo_q, args.hi_q)

    if not args.apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply to commit.")
        return
    apply(ego_root, ego_files, plan, args.lo_q, args.hi_q)
    print("\nDONE. Norm stats are now STALE -- recompute before training:")
    print("  rm -rf assets/pi05_yam7h_ea/yam7h_p50")
    print("  uv run scripts/compute_norm_stats.py --config-name pi05_yam7h_ea --max-frames 200000 --skip-videos")


if __name__ == "__main__":
    main()

"""Per-source mixture diagnostics: what teleop and ego actually teach.

Parquet only -- no video decode -- so it runs in a couple of minutes.

Motivation. vast_run/preflight_checks.py section [2] reports
`max |action[t] - state[t+1]|`, an infinity norm over sampled episodes. That is
enough to classify a source (ego is exactly 0, teleop is not) but it is the wrong
summary for "how much corrective signal is in teleop", because a max is set by the
single worst frame in the sample. This script reports the distribution instead, plus
the two things that behave differently under flow matching than under FAST-token CE.

  [A] Tracking residual   r[t] = action[t] - state[t+1], per dimension.
                          Exactly 0 for ego by construction. For teleop, the median
                          and the LEAD RATIO -- |r| as a fraction of the per-step
                          increment -- are what say how much of each command is lead
                          rather than just the next state. After
                          DeltaActions(6,-1,6,-1) the target is
                          (state[t+k+1] - state[t]) + r[t+k]; the residual is the
                          entire difference between the two sources' supervision.

  [B] Normalized action   pi0.5 regresses continuous actions, so where each source
      distribution         lands on the [-1,1] normalized axis IS its gradient
                          contribution. Narrow ego deltas + mixture-wide quantile
                          norm means ego targets sit near 0, where the flow target
                          (noise - action) is dominated by the noise term. Ego's
                          effective share is then lower than its sampling share.

  [C] Clip atoms          The retargeting clips arm deltas at +/-0.1 and +/-0.2.
                          If those atoms sit near q01/q99, the normalization
                          constants are being set by an artifact.

  [D] Gripper collision    Where each source's open/closed modes land after
                          normalization. Widely separated modes for the same
                          semantic command are the mode-averaging risk under
                          few-step flow sampling.

Usage (from the openpi repo root):
    uv run vast_run/pi05/mixture_diagnostics.py
    uv run vast_run/pi05/mixture_diagnostics.py \
        --norm-stats assets/pi05_yam7h_ea/yam7h_p50/norm_stats.json \
        --episodes 40 --out /workspace/diagnostics
"""

import argparse
import glob
import json
import os
import pathlib

import numpy as np
import pyarrow.parquet as pq

# State/action layout: [R_j1-6, R_grip, L_j1-6, L_grip]  (matches preflight_checks.py)
GRIPPER_DIMS = (6, 13)
ARM_DIMS = [d for d in range(14) if d not in GRIPPER_DIMS]
DIM_NAMES = [f"R_j{i + 1}" for i in range(6)] + ["R_grip"] + [f"L_j{i + 1}" for i in range(6)] + ["L_grip"]

# openpi applies DeltaActions(make_bool_mask(6, -1, 6, -1)): arm dims relative to the
# current state, grippers left absolute.
DELTA_MASK = np.array([d in ARM_DIMS for d in range(14)])

DEFAULT_ROOTS = {
    "teleop": os.environ.get("YAM7H_TELEOP_ROOT", "/workspace/data/yam7h/teleop"),
    "ego": os.environ.get("YAM7H_EGO_ROOT", "/workspace/data/yam7h/ego"),
}
CLIP_VALUES = (0.1, 0.2)


# --------------------------------------------------------------------------------
def load_episodes(root: pathlib.Path, max_episodes: int):
    """Yield (state[T,14], action[T,14]) from a spread of parquet files."""
    files = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"no parquet under {root}/data -- is the dataset present?")
    step = max(1, len(files) // max_episodes)
    for path in files[::step][:max_episodes]:
        table = pq.read_table(path, columns=["observation.state", "action"])
        state = np.stack(table["observation.state"].to_numpy(zero_copy_only=False)).astype(np.float64)
        action = np.stack(table["action"].to_numpy(zero_copy_only=False)).astype(np.float64)
        if state.shape[1] != 14 or action.shape[1] != 14:
            raise SystemExit(f"{path}: expected 14-dim state/action, got {state.shape}/{action.shape}")
        yield state, action


def normalize_q(x, q01, q99):
    """Exactly openpi's transforms.Normalize._normalize_quantile."""
    return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


# --------------------------------------------------------------------------------
# [A] tracking residual
# --------------------------------------------------------------------------------
def residual_report(episodes) -> dict:
    resid, incr = [], []
    for state, action in episodes:
        if len(state) < 2:
            continue
        # r[t] = action[t] - state[t+1]; ego satisfies this exactly == 0.
        resid.append(action[:-1] - state[1:])
        incr.append(state[1:] - state[:-1])
    r = np.concatenate(resid)
    d = np.concatenate(incr)

    out = {}
    for dim in range(14):
        rr, dd = r[:, dim], d[:, dim]
        med_r = float(np.median(np.abs(rr)))
        med_d = float(np.median(np.abs(dd)))
        # Signed lead: positive means the command sits AHEAD of the achieved state in
        # the direction the joint is travelling -- that is the corrective signal.
        signed = float(np.median(rr * np.sign(dd)))
        out[dim] = {
            "median_abs_r": med_r,
            "p99_abs_r": float(np.percentile(np.abs(rr), 99)),
            "max_abs_r": float(np.abs(rr).max()),
            "median_abs_increment": med_d,
            "lead_ratio": med_r / med_d if med_d > 1e-12 else float("nan"),
            "signed_lead": signed,
        }
    return out


def print_residual(name: str, stats: dict) -> None:
    print(f"\n  --- {name} ---")
    print(f"  {'dim':8s} {'med|r|':>10s} {'p99|r|':>10s} {'max|r|':>10s} "
          f"{'med|Δstate|':>12s} {'lead ratio':>11s} {'signed lead':>12s}")
    for dim in range(14):
        s = stats[dim]
        print(f"  {DIM_NAMES[dim]:8s} {s['median_abs_r']:10.5f} {s['p99_abs_r']:10.5f} "
              f"{s['max_abs_r']:10.5f} {s['median_abs_increment']:12.5f} "
              f"{s['lead_ratio']:11.3f} {s['signed_lead']:+12.5f}")
    arm_lead = np.nanmedian([stats[d]["lead_ratio"] for d in ARM_DIMS])
    print(f"  arm-dim median lead ratio: {arm_lead:.3f}")


# --------------------------------------------------------------------------------
# [B]/[C] normalized action chunks and clip atoms
# --------------------------------------------------------------------------------
def action_report(episodes, q01, q99, horizon: int, stride: int) -> tuple[dict, dict]:
    chunks, one_step = [], []
    for state, action in episodes:
        n_frames = len(state)
        if n_frames < 2:
            continue
        one_step.append(action - state)  # raw per-step delta, before the mask
        starts = np.arange(0, n_frames, stride)
        for t in starts:
            idx = np.minimum(np.arange(t, t + horizon), n_frames - 1)  # clamp at episode end
            chunk = action[idx].copy()
            chunk[:, DELTA_MASK] -= state[t][DELTA_MASK]  # DeltaActions
            chunks.append(chunk)
    chunk_arr = np.concatenate(chunks) if chunks else np.empty((0, 14))
    raw_delta = np.concatenate(one_step)

    norm = normalize_q(chunk_arr, q01, q99)

    stats = {}
    for dim in range(14):
        col = norm[:, dim]
        stats[dim] = {
            "p01": float(np.percentile(col, 1)),
            "p50": float(np.percentile(col, 50)),
            "p99": float(np.percentile(col, 99)),
            "std": float(col.std()),
            "frac_within_0.1": float(np.mean(np.abs(col) < 0.1)),
            "frac_clipped_to_range": float(np.mean(np.abs(col) > 1.0)),
        }

    atoms = {}
    for dim in ARM_DIMS:
        col = raw_delta[:, dim]
        per_dim = {}
        for v in CLIP_VALUES:
            for sign in (+1, -1):
                target = sign * v
                frac = float(np.mean(np.isclose(col, target, atol=1e-6)))
                if frac > 1e-4:
                    per_dim[f"{target:+.1f}"] = {
                        "fraction": frac,
                        "normalized": float(normalize_q(np.array(target), q01[dim], q99[dim])),
                    }
        if per_dim:
            atoms[dim] = per_dim
    return stats, atoms


def print_action(name: str, stats: dict) -> None:
    print(f"\n  --- {name} (normalized action chunks) ---")
    print(f"  {'dim':8s} {'p01':>9s} {'p50':>9s} {'p99':>9s} {'std':>9s} "
          f"{'|x|<0.1':>9s} {'|x|>1':>9s}")
    for dim in range(14):
        s = stats[dim]
        print(f"  {DIM_NAMES[dim]:8s} {s['p01']:9.4f} {s['p50']:9.4f} {s['p99']:9.4f} "
              f"{s['std']:9.4f} {s['frac_within_0.1']:8.1%} {s['frac_clipped_to_range']:8.1%}")


# --------------------------------------------------------------------------------
# [D] gripper modes on the normalized axis
# --------------------------------------------------------------------------------
def gripper_modes(episodes_states, q01, q99) -> dict:
    """Two-mode split per gripper dim (median of each half around the midrange)."""
    out = {}
    for dim in GRIPPER_DIMS:
        col = episodes_states[:, dim]
        lo, hi = np.percentile(col, [1, 99])
        mid = 0.5 * (lo + hi)
        low, high = col[col <= mid], col[col > mid]
        closed = float(np.median(low)) if low.size else float("nan")
        opened = float(np.median(high)) if high.size else float("nan")
        out[dim] = {
            "closed_raw": closed,
            "open_raw": opened,
            "closed_norm": float(normalize_q(np.array(closed), q01[dim], q99[dim])),
            "open_norm": float(normalize_q(np.array(opened), q01[dim], q99[dim])),
            "frac_low_mode": float(low.size / col.size),
        }
    return out


# --------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teleop-root", default=DEFAULT_ROOTS["teleop"])
    ap.add_argument("--ego-root", default=DEFAULT_ROOTS["ego"])
    ap.add_argument("--norm-stats", default="assets/pi05_yam7h_ea/yam7h_p50/norm_stats.json")
    ap.add_argument("--episodes", type=int, default=30, help="parquet files sampled per source")
    ap.add_argument("--horizon", type=int, default=50, help="action_horizon used by the config")
    ap.add_argument("--stride", type=int, default=10, help="subsample chunk start frames")
    ap.add_argument("--out", default="/workspace/diagnostics")
    args = ap.parse_args()

    ns_path = pathlib.Path(args.norm_stats)
    if not ns_path.is_file():
        raise SystemExit(
            f"norm stats not found at {ns_path}.\n"
            "Point --norm-stats at the arm you are about to run, e.g.\n"
            "  assets/pi0_fast_yam7h_ea/yam7h_p50/norm_stats.json"
        )
    ns = json.loads(ns_path.read_text())["norm_stats"]
    aq01 = np.asarray(ns["actions"]["q01"], dtype=np.float64)
    aq99 = np.asarray(ns["actions"]["q99"], dtype=np.float64)

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    roots = {"teleop": pathlib.Path(args.teleop_root), "ego": pathlib.Path(args.ego_root)}
    print(f"norm stats : {ns_path}")
    print(f"horizon    : {args.horizon}   chunk stride: {args.stride}   episodes/source: {args.episodes}")

    summary: dict = {"norm_stats": str(ns_path), "horizon": args.horizon, "sources": {}}

    print("\n=== [A] TRACKING RESIDUAL  r[t] = action[t] - state[t+1] ===")
    print("  lead ratio = median|r| / median|state[t+1]-state[t]|  (0.0 means action IS the next state)")
    print("  signed lead > 0 means the command sits ahead of the achieved state, i.e. real corrective signal")
    residuals, actions, grippers, all_states = {}, {}, {}, {}
    for name, root in roots.items():
        eps = list(load_episodes(root, args.episodes))
        residuals[name] = residual_report(eps)
        print_residual(name, residuals[name])
        actions[name], atoms = action_report(eps, aq01, aq99, args.horizon, args.stride)
        all_states[name] = np.concatenate([s for s, _ in eps])
        grippers[name] = gripper_modes(all_states[name], aq01, aq99)
        summary["sources"][name] = {
            "root": str(root),
            "residual": {DIM_NAMES[k]: v for k, v in residuals[name].items()},
            "normalized_actions": {DIM_NAMES[k]: v for k, v in actions[name].items()},
            "clip_atoms": {DIM_NAMES[k]: v for k, v in atoms.items()},
            "gripper_modes": {DIM_NAMES[k]: v for k, v in grippers[name].items()},
        }
        summary["sources"][name]["_atoms_tmp"] = atoms

    print("\n=== [B] NORMALIZED ACTION DISTRIBUTION (what flow matching actually regresses) ===")
    for name in roots:
        print_action(name, actions[name])
    t_std = float(np.median([actions["teleop"][d]["std"] for d in ARM_DIMS]))
    e_std = float(np.median([actions["ego"][d]["std"] for d in ARM_DIMS]))
    print(f"\n  median arm-dim std, teleop {t_std:.4f} vs ego {e_std:.4f}  (ratio {t_std / max(e_std, 1e-9):.2f}x)")
    print("  A large ratio means ego's EFFECTIVE gradient share is below its sampling share:")
    print("  the flow target is (noise - action), so near-zero actions leave mostly noise to regress.")

    print("\n=== [C] CLIP ATOMS IN THE RAW PER-STEP DELTA ===")
    any_atoms = False
    for name in roots:
        atoms = summary["sources"][name].pop("_atoms_tmp")
        if not atoms:
            print(f"  {name}: none detected at +/-0.1 or +/-0.2")
            continue
        any_atoms = True
        print(f"  {name}:")
        for dim, per_dim in atoms.items():
            for val, info in per_dim.items():
                edge = "AT/BEYOND the normalization edge" if abs(info["normalized"]) > 0.9 else "inside range"
                print(f"    {DIM_NAMES[dim]:8s} {val:>5s} raw -> {info['normalized']:+.3f} normalized "
                      f"({info['fraction']:.2%} of frames, {edge})")
    if any_atoms:
        print("  Atoms landing near +/-1.0 mean q01/q99 are being set by a retargeting artifact,")
        print("  not by the real action distribution. Consider fitting norm stats on teleop only.")

    print("\n=== [D] GRIPPER MODES ON THE NORMALIZED AXIS ===")
    for dim in GRIPPER_DIMS:
        print(f"  {DIM_NAMES[dim]}:")
        pts = []
        for name in roots:
            g = grippers[name][dim]
            print(f"    {name:7s} closed {g['closed_raw']:.3f} -> {g['closed_norm']:+.3f} | "
                  f"open {g['open_raw']:.3f} -> {g['open_norm']:+.3f} "
                  f"({g['frac_low_mode']:.0%} of frames in the low mode)")
            pts.append((f"{name}-open", g["open_norm"]))
        gap = abs(pts[0][1] - pts[1][1])
        if gap > 0.5:
            print(f"    *** 'open' differs by {gap:.2f} on the normalized axis between sources. ***")
            print("        Few-step flow sampling can land BETWEEN two far-apart modes, which on a")
            print("        real gripper is a partially-closed hand. Rescale the ego gripper channel")
            print("        onto teleop's range in build_mixture.py, then recompute norm stats.")

    (out_dir / "mixture_diagnostics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsummary -> {out_dir}/mixture_diagnostics.json")


if __name__ == "__main__":
    main()

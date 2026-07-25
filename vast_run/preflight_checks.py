"""BLOCKING pre-flight checks for the teleop/ego co-fine-tune (spec section 2).

Reads parquet only -- no video decode -- so it runs in a couple of minutes over
both datasets.

  1. Gripper convention: do "open" and "closed" map to the same numeric direction
     in teleop and ego? An inverted convention silently trains contradictory
     gripper supervision and confounds every downstream comparison.
  2. Delta distributions: after the absolute->delta transform openpi applies to
     arm joints, per-dimension deltas from both sources should broadly overlap.
     A large systematic offset means a convention mismatch upstream.
  3. Action convention: ego is expected to satisfy action[t] == state[t+1]
     exactly (zero corrective signal); teleop should not.

Usage:
    uv run vast_run/preflight_checks.py
    uv run vast_run/preflight_checks.py --episodes 40 --out /workspace/preflight
"""

import argparse
import glob
import json
import os
import pathlib

import numpy as np
import pyarrow.parquet as pq

REPOS = {
    "teleop": "angkul07/abc-teleop",
    "ego": "angkul07/EgoDex-PickPlace-YAM-14dof-multiview",
}
LEROBOT_HOME = os.environ.get("HF_LEROBOT_HOME", "/workspace/.hf_home/lerobot")

# State/action layout: [R_j1-6, R_grip, L_j1-6, L_grip]
GRIPPER_DIMS = (6, 13)
ARM_DIMS = [d for d in range(14) if d not in GRIPPER_DIMS]
DIM_NAMES = [f"R_j{i + 1}" for i in range(6)] + ["R_grip"] + [f"L_j{i + 1}" for i in range(6)] + ["L_grip"]


def load_episodes(root: pathlib.Path, max_episodes: int):
    """Yield (episode_index, state[T,14], action[T,14]) from parquet files."""
    files = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"no parquet under {root}/data -- is the dataset downloaded?")
    step = max(1, len(files) // max_episodes)
    for i, path in enumerate(files[::step][:max_episodes]):
        table = pq.read_table(path, columns=["observation.state", "action"])
        state = np.stack(table["observation.state"].to_numpy(zero_copy_only=False)).astype(np.float64)
        action = np.stack(table["action"].to_numpy(zero_copy_only=False)).astype(np.float64)
        if state.shape[1] != 14 or action.shape[1] != 14:
            raise SystemExit(f"{path}: expected 14-dim state/action, got {state.shape}/{action.shape}")
        yield i, state, action


def collect(name: str, max_episodes: int):
    root = pathlib.Path(LEROBOT_HOME) / REPOS[name]
    states, actions, first_frames, next_state_gap = [], [], [], []
    n_eps = 0
    for _, state, action in load_episodes(root, max_episodes):
        states.append(state)
        actions.append(action)
        first_frames.append(state[0])
        if len(state) > 1:
            # ego is expected to satisfy action[t] == state[t+1] exactly
            next_state_gap.append(np.abs(action[:-1] - state[1:]).max())
        n_eps += 1
    return {
        "name": name,
        "root": str(root),
        "episodes": n_eps,
        "state": np.concatenate(states),
        "action": np.concatenate(actions),
        "first_frames": np.stack(first_frames),
        "next_state_gap": float(np.max(next_state_gap)) if next_state_gap else float("nan"),
    }


def pct(x, q):
    return float(np.percentile(x, q))


def gripper_report(data):
    print(f"\n--- {data['name']}  ({data['episodes']} episodes, {len(data['state']):,} frames) ---")
    rows = {}
    for dim in GRIPPER_DIMS:
        g = data["state"][:, dim]
        start = data["first_frames"][:, dim]
        # Fraction of time in the bottom/top 10% of the [0,1] range.
        low = float(np.mean(g < 0.1))
        high = float(np.mean(g > 0.9))
        rows[dim] = {
            "mean": float(g.mean()),
            "min": float(g.min()),
            "max": float(g.max()),
            "p01": pct(g, 1),
            "p50": pct(g, 50),
            "p99": pct(g, 99),
            "frac_below_0.1": low,
            "frac_above_0.9": high,
            "episode_start_mean": float(start.mean()),
        }
        print(
            f"  {DIM_NAMES[dim]:7s} mean={rows[dim]['mean']:.3f}  "
            f"range=[{rows[dim]['min']:.3f}, {rows[dim]['max']:.3f}]  "
            f"p01/p50/p99={rows[dim]['p01']:.2f}/{rows[dim]['p50']:.2f}/{rows[dim]['p99']:.2f}  "
            f"time_low={low:.0%} time_high={high:.0%}  ep_start_mean={rows[dim]['episode_start_mean']:.3f}"
        )
    return rows


def delta_report(data):
    """Per-dim stats of the delta targets openpi actually trains on."""
    state, action = data["state"], data["action"]
    delta = action.copy()
    delta[:, ARM_DIMS] -= state[:, ARM_DIMS]  # grippers stay absolute (delta mask 6,-1,6,-1)
    out = {}
    for dim in range(14):
        col = delta[:, dim]
        out[dim] = {"mean": float(col.mean()), "std": float(col.std()), "p01": pct(col, 1), "p99": pct(col, 99)}
    return out, delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30, help="episodes sampled per dataset")
    ap.add_argument("--out", default="/workspace/preflight", help="directory for plots + json")
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== [0] SCHEMA MATCH (both sources share one transform chain) ===")
    infos = {}
    for name, repo in REPOS.items():
        info_path = pathlib.Path(LEROBOT_HOME) / repo / "meta" / "info.json"
        if not info_path.exists():
            raise SystemExit(f"{info_path} missing -- run vast_run/dl.py first")
        infos[name] = json.loads(info_path.read_text())
        i = infos[name]
        print(
            f"  {name:7s} v{i.get('codebase_version')}  fps={i.get('fps')}  "
            f"episodes={i.get('total_episodes')}  frames={i.get('total_frames'):,}"
        )
    keys = {n: sorted(i["features"]) for n, i in infos.items()}
    if keys["teleop"] != keys["ego"]:
        only_t = set(keys["teleop"]) - set(keys["ego"])
        only_e = set(keys["ego"]) - set(keys["teleop"])
        raise SystemExit(f"FEATURE KEYS DIFFER -- teleop only: {sorted(only_t)}, ego only: {sorted(only_e)}")
    for key in ("observation.images.top", "observation.images.left_wrist", "observation.images.right_wrist",
                "observation.state", "action"):
        if key not in keys["teleop"]:
            raise SystemExit(f"missing expected feature '{key}' (repack transform would KeyError)")
        shapes = {n: infos[n]["features"][key]["shape"] for n in REPOS}
        if shapes["teleop"] != shapes["ego"]:
            raise SystemExit(f"shape mismatch for '{key}': {shapes}")
    if infos["teleop"]["fps"] != infos["ego"]["fps"]:
        raise SystemExit(f"fps mismatch: {infos['teleop']['fps']} vs {infos['ego']['fps']} "
                         "-- action chunks would span different durations")
    print("  feature keys, shapes and fps match")

    data = {name: collect(name, args.episodes) for name in REPOS}

    print("\n=== [1] GRIPPER CONVENTION (spec 2.1 -- BLOCKING) ===")
    grippers = {name: gripper_report(d) for name, d in data.items()}

    print("\n  verdict (heuristic -- confirm by eye on the plot before training):")
    flagged = False
    for dim in GRIPPER_DIMS:
        t, e = grippers["teleop"][dim], grippers["ego"][dim]
        # If one source sits mostly high while the other sits mostly low, and their
        # episode starts disagree in the same direction, the convention is inverted.
        gap = t["mean"] - e["mean"]
        start_gap = t["episode_start_mean"] - e["episode_start_mean"]
        inverted = abs(gap) > 0.3 and np.sign(gap) == np.sign(start_gap) and abs(start_gap) > 0.3
        status = "LIKELY INVERTED" if inverted else "consistent-ish"
        flagged |= inverted
        print(
            f"    {DIM_NAMES[dim]:7s} teleop_mean={t['mean']:.3f} ego_mean={e['mean']:.3f} "
            f"(gap {gap:+.3f}, episode-start gap {start_gap:+.3f}) -> {status}"
        )
    print(
        "    NOTE: a mean gap alone is NOT proof of inversion (ego may simply grasp less "
        "often).\n          The decisive evidence is the plotted traces around grasp events."
    )

    print("\n=== [2] ACTION CONVENTION (spec section 1) ===")
    for name, d in data.items():
        gap = d["next_state_gap"]
        kind = "action[t] == state[t+1] (no corrective signal)" if gap < 1e-6 else "commanded targets"
        print(f"  {name:7s} max|action[t] - state[t+1]| = {gap:.2e}  ->  {kind}")

    print("\n=== [3] DELTA DISTRIBUTIONS (spec 2.2) ===")
    deltas = {}
    stats = {}
    for name, d in data.items():
        stats[name], deltas[name] = delta_report(d)
    print(f"  {'dim':8s} {'teleop mean':>12s} {'ego mean':>12s} {'teleop p01/p99':>22s} {'ego p01/p99':>22s}")
    for dim in range(14):
        t, e = stats["teleop"][dim], stats["ego"][dim]
        print(
            f"  {DIM_NAMES[dim]:8s} {t['mean']:12.5f} {e['mean']:12.5f} "
            f"{t['p01']:10.4f}/{t['p99']:<10.4f} {e['p01']:10.4f}/{e['p99']:<10.4f}"
        )
    print("  (arm dims are deltas; R_grip/L_grip are absolute -- they are not expected to match in scale)")

    # ---- plots ----
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        for col, dim in enumerate(GRIPPER_DIMS):
            ax = axes[0][col]
            for name, d in data.items():
                ax.hist(d["state"][:, dim], bins=60, alpha=0.5, density=True, label=name)
            ax.set_title(f"{DIM_NAMES[dim]} value distribution")
            ax.legend()
            ax = axes[1][col]
            # first episode's gripper trace: the shape around grasp/release is the
            # thing a human actually has to eyeball
            for name, d in data.items():
                trace = d["state"][:1500, dim]
                ax.plot(trace, label=f"{name} (first frames)", alpha=0.8)
            ax.set_title(f"{DIM_NAMES[dim]} trace")
            ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "gripper_convention.png", dpi=110)

        fig, axes = plt.subplots(3, 4, figsize=(16, 9))
        for ax, dim in zip(axes.flat, ARM_DIMS):
            for name in REPOS:
                col = deltas[name][:, dim]
                lo, hi = np.percentile(np.concatenate([deltas[n][:, dim] for n in REPOS]), [0.5, 99.5])
                ax.hist(col, bins=80, range=(lo, hi), alpha=0.5, density=True, label=name)
            ax.set_title(DIM_NAMES[dim])
        axes.flat[0].legend()
        fig.suptitle("Per-dimension action-delta distributions (should broadly overlap)")
        fig.tight_layout()
        fig.savefig(out_dir / "delta_distributions.png", dpi=110)
        print(f"\nplots -> {out_dir}/gripper_convention.png, {out_dir}/delta_distributions.png")
    except ImportError:
        print("\n(matplotlib not available -- skipped plots)")

    summary = {
        "episodes_per_source": args.episodes,
        "gripper": {n: {str(k): v for k, v in g.items()} for n, g in grippers.items()},
        "next_state_gap": {n: d["next_state_gap"] for n, d in data.items()},
        "delta_stats": {n: {DIM_NAMES[k]: v for k, v in s.items()} for n, s in stats.items()},
    }
    (out_dir / "preflight.json").write_text(json.dumps(summary, indent=2))
    print(f"summary -> {out_dir}/preflight.json")

    if flagged:
        print("\n*** GRIPPER CONVENTION FLAGGED -- inspect the plot before training. ***")
        raise SystemExit(2)
    print("\nPRE-FLIGHT COMPLETE (still eyeball gripper_convention.png before launching).")


if __name__ == "__main__":
    main()

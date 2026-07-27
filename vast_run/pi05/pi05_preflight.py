"""Pre-launch checks specific to the pi0-FAST -> pi0.5 swap.

Everything here catches a failure that would otherwise surface late, expensively, or
silently:

  1. Config shape       -- action_dim must be 32 (weight_loaders._merge_params matches
                           key names only and NEVER compares shapes, so a mismatch does
                           not raise at load; it dies inside jit with an unrelated
                           message). discrete_state_input must be True, or the model
                           gets no proprioception at all.
  2. Trainable split    -- confirms the 311M action expert is trainable and SigLIP is
                           unfrozen (the freeze regex is ".*llm.*"; SigLIP lives at
                           PaliGemma.img, so it is trainable in BOTH families).
  3. Prompt length      -- pi0.5's prompt is "Task: {task}, State: {ints};\\nAction: ".
                           If it exceeds max_token_len the trailing "Action: " is
                           truncated away, which is quietly destructive. Measured on
                           your real task strings and real normalized states.
  4. Norm stats         -- present, 14-dim, and carry q01/q99 (quantile norm is on for
                           PI05, so mean/std alone would abort at _assert_quantile_stats).

Usage (from the openpi repo root):
    uv run vast_run/pi05/pi05_preflight.py pi05_yam7h_ea
    uv run vast_run/pi05/pi05_preflight.py pi05_yam7h_ea --episodes 12

Exit code 0 = clear to launch, 2 = something needs fixing.
"""

import argparse
import glob
import json
import pathlib
import sys

import numpy as np

FAILURES: list[str] = []
WARNINGS: list[str] = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"  FAIL  {msg}")


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"  WARN  {msg}")


def ok(msg: str) -> None:
    print(f"  ok    {msg}")


# --------------------------------------------------------------------------------
# [1] config shape
# --------------------------------------------------------------------------------
def check_config(cfg):
    print("\n=== [1] CONFIG SHAPE ===")
    m = cfg.model
    mt = m.model_type.name
    print(f"  {type(m).__name__}  model_type={mt}  action_dim={m.action_dim}  "
          f"horizon={m.action_horizon}  max_token_len={m.max_token_len}")

    if mt != "PI05":
        fail(f"model_type is {mt}, expected PI05 -- is pi05=True set on Pi0Config?")
        return

    if m.action_dim != 32:
        fail(
            f"action_dim={m.action_dim}, must be 32. pi05_base ships "
            f"action_in_proj=Linear(32,1024) / action_out_proj=Linear(1024,32). "
            f"_merge_params does not check shapes, so this will not raise at load."
        )
    else:
        ok("action_dim=32 matches pi05_base's action projections")

    if not getattr(m, "discrete_state_input", True):
        fail(
            "discrete_state_input=False with pi05=True -> the model receives NO state. "
            "embed_suffix skips the state token when pi05 is set, so the discretized "
            "prompt is the only path in. Leave the field unset."
        )
    else:
        ok("discrete_state_input=True (state reaches the model via the prompt)")

    if m.paligemma_variant != "gemma_2b_lora":
        warn(f"paligemma_variant={m.paligemma_variant!r} -- expected gemma_2b_lora for the LoRA arms")

    if m.action_horizon != 50:
        warn(f"action_horizon={m.action_horizon}, the pi0-FAST arms used 50 (1.67 s at 30 Hz)")

    sched = cfg.lr_schedule
    print(f"  schedule: warmup={sched.warmup_steps} peak={sched.peak_lr} "
          f"decay_steps={sched.decay_steps} decay_lr={sched.decay_lr} steps={cfg.num_train_steps}")
    if sched.decay_steps != cfg.num_train_steps:
        fail("decay_steps != num_train_steps (run_yam.sh asserts this too)")


# --------------------------------------------------------------------------------
# [2] trainable / frozen split
# --------------------------------------------------------------------------------
def _bucket(path: str) -> str:
    if "lora" in path:
        return "LoRA adapters"
    if "/img/" in path or path.startswith("PaliGemma/img"):
        return "SigLIP vision tower"
    if "_1" in path:
        return "action expert (gemma_300m)"
    if "action_in_proj" in path or "action_out_proj" in path or "time_mlp" in path:
        return "action projections / time MLP"
    if "llm" in path:
        return "Gemma 2B trunk"
    return "other"


def check_params(cfg):
    print("\n=== [2] TRAINABLE / FROZEN SPLIT ===")
    try:
        from flax import nnx  # noqa: PLC0415  (heavy import, diagnostic only)
        import jax  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover
        warn(f"could not import jax/flax ({e}) -- skipping param accounting")
        return

    try:
        state = jax.eval_shape(lambda rng: nnx.state(cfg.model.create(rng)), jax.random.key(0))
        trainable = state.filter(cfg.trainable_filter)
        frozen = state.filter(cfg.freeze_filter)
    except Exception as e:  # purely diagnostic, never block on it
        warn(f"param accounting failed ({type(e).__name__}: {e}) -- skipping")
        return

    def tally(tree):
        out: dict[str, int] = {}
        total = 0
        for keypath, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
            path = "/".join(str(getattr(k, "key", k)) for k in keypath)
            n = int(np.prod(leaf.shape)) if hasattr(leaf, "shape") else 0
            out[_bucket(path)] = out.get(_bucket(path), 0) + n
            total += n
        return out, total

    tr, tr_total = tally(trainable)
    fr, fr_total = tally(frozen)

    print(f"  trainable: {tr_total / 1e6:8.1f}M")
    for k, v in sorted(tr.items(), key=lambda kv: -kv[1]):
        print(f"      {k:32s} {v / 1e6:8.1f}M")
    print(f"  frozen   : {fr_total / 1e6:8.1f}M")
    for k, v in sorted(fr.items(), key=lambda kv: -kv[1]):
        print(f"      {k:32s} {v / 1e6:8.1f}M")

    if not any("action expert" in k for k in tr):
        fail("action expert is NOT in the trainable set -- check the freeze_filter arguments")
    else:
        ok("action expert is trainable")
    if not any("SigLIP" in k for k in tr):
        warn("SigLIP is frozen -- unexpected for this repo (freeze regex is '.*llm.*')")
    else:
        ok("SigLIP vision tower is trainable (as in the pi0-FAST arms)")

    # AdamW carries m + v in fp32 for the trainable set, replicated per GPU under
    # --fsdp-devices 1. Grads too. Rough, but the right order of magnitude.
    print(f"  ~optimizer + grad memory per GPU: {tr_total * 12 / 1e9:.1f} GB")


# --------------------------------------------------------------------------------
# [3] prompt token length on real data
# --------------------------------------------------------------------------------
def _load_tasks(root: pathlib.Path) -> list[str]:
    f = root / "meta" / "tasks.jsonl"
    if not f.is_file():
        return []
    tasks = []
    for raw_line in f.read_text().splitlines():
        line = raw_line.strip()
        if line:
            tasks.append(json.loads(line)["task"])
    return tasks


def _load_states(root: pathlib.Path, n_files: int) -> np.ndarray:
    import pyarrow.parquet as pq  # noqa: PLC0415

    files = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        return np.empty((0, 14))
    step = max(1, len(files) // n_files)
    chunks = []
    for path in files[::step][:n_files]:
        t = pq.read_table(path, columns=["observation.state"])
        chunks.append(np.stack(t["observation.state"].to_numpy(zero_copy_only=False)).astype(np.float64))
    return np.concatenate(chunks) if chunks else np.empty((0, 14))


def check_prompt_length(cfg, data_cfg, norm_stats, episodes: int):
    print("\n=== [3] PROMPT TOKEN LENGTH (real tasks x real states) ===")
    max_len = cfg.model.max_token_len
    try:
        from openpi.models import tokenizer as _tokenizer  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover
        warn(f"could not import the tokenizer ({e}) -- skipping")
        return

    if norm_stats is None:
        warn("no norm stats -- cannot normalize states, skipping")
        return

    q01 = np.asarray(norm_stats["state"]["q01"], dtype=np.float64)
    q99 = np.asarray(norm_stats["state"]["q99"], dtype=np.float64)

    # A deliberately huge max_len so mask.sum() reports the TRUE untruncated length.
    probe = _tokenizer.PaligemmaTokenizer(4096)

    import lerobot.common.datasets.lerobot_dataset as _lerobot  # noqa: F401, PLC0415  (env parity)

    lengths: list[int] = []
    for src in data_cfg.mixture:
        root = pathlib.Path(src.root) if src.root else None
        if root is None or not root.is_dir():
            warn(f"{src.repo_id}: root not found locally, skipping prompt check")
            continue
        tasks = _load_tasks(root)
        states = _load_states(root, episodes)
        if not tasks or states.size == 0:
            warn(f"{src.repo_id}: no tasks.jsonl or no parquet found, skipping")
            continue

        norm = (states - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(norm), size=min(400, len(norm)))
        src_lengths = []
        for i in idx:
            task = tasks[int(rng.integers(0, len(tasks)))]
            _, mask = probe.tokenize(task, norm[i])
            src_lengths.append(int(mask.sum()))
        lengths.extend(src_lengths)
        a = np.asarray(src_lengths)
        print(f"  {src.repo_id:52s} n={len(a):4d}  p50={np.percentile(a, 50):5.0f}  "
              f"p99={np.percentile(a, 99):5.0f}  max={a.max():5.0f}   "
              f"({len(tasks)} distinct task strings)")

    if not lengths:
        warn("no prompt lengths measured")
        return

    arr = np.asarray(lengths)
    headroom = max_len / arr.max()
    print(f"  configured max_token_len = {max_len}, observed max = {arr.max()}, headroom = {headroom:.2f}x")
    if arr.max() >= max_len:
        fail(
            f"prompts reach {arr.max()} tokens >= max_token_len={max_len}. The trailing "
            f"'Action: ' marker gets truncated. Raise max_token_len."
        )
    elif headroom < 1.25:
        warn(f"only {headroom:.2f}x headroom -- raise max_token_len to ~{int(arr.max() * 1.5)} to be safe")
    else:
        ok(f"comfortable headroom ({headroom:.2f}x)")


# --------------------------------------------------------------------------------
# [4] norm stats
# --------------------------------------------------------------------------------
def check_norm_stats(cfg, data_cfg):
    print("\n=== [4] NORM STATS ===")
    path = pathlib.Path("assets") / cfg.name / data_cfg.asset_id / "norm_stats.json"
    print(f"  expected at {path}")
    if not path.is_file():
        fast_twin = str(path).replace("pi05_yam7h_", "pi0_fast_yam7h_")
        fail(
            f"missing. They are identical to the pi0-FAST arm's -- copy them:\n"
            f"          mkdir -p {path.parent}\n"
            f"          cp -r {pathlib.Path(fast_twin).parent}/. {path.parent}/"
        )
        return None

    raw = json.loads(path.read_text())["norm_stats"]
    for key in ("state", "actions"):
        if key not in raw:
            fail(f"norm stats missing '{key}'")
            return None
        entry = raw[key]
        if "q01" not in entry or "q99" not in entry:
            fail(f"'{key}' has no q01/q99 -- PI05 uses quantile norm and will abort at _assert_quantile_stats")
            return None
        n = len(entry["q01"])
        if n != 14:
            fail(f"'{key}' q01 is {n}-dim, expected 14 (stats are computed pre-padding)")
            return None
        ok(f"'{key}': 14-dim with q01/q99")
    print(f"  use_quantile_norm = {data_cfg.use_quantile_norm} (True for both PI0_FAST and PI05)")
    return raw


# --------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="e.g. pi05_yam7h_ea")
    ap.add_argument("--episodes", type=int, default=8, help="parquet files sampled per source for the state pool")
    ap.add_argument("--skip-params", action="store_true", help="skip the jax.eval_shape param accounting")
    args = ap.parse_args()

    import openpi.training.config as _config  # noqa: PLC0415  (after argparse, so --help is instant)

    cfg = _config.get_config(args.config)
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)

    print(f"pi0.5 preflight for {cfg.name}")
    check_config(cfg)
    if not args.skip_params:
        check_params(cfg)
    norm_stats = check_norm_stats(cfg, data_cfg)
    check_prompt_length(cfg, data_cfg, norm_stats, args.episodes)

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} BLOCKING ISSUE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(2)
    if WARNINGS:
        print(f"{len(WARNINGS)} warning(s) -- readable above, none blocking.")
    print("PREFLIGHT CLEAR -- safe to launch.")


if __name__ == "__main__":
    main()

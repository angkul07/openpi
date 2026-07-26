"""Independent audit: does the rebuilt data actually match its source?

build_mixture.py's own verify() checks structure (contiguous indices, files present,
counts matching). It does NOT check that the payload survived renumbering, that videos
decode to the right number of frames, or that the holdout stayed excluded. Those are
the things that would silently poison training, so check them separately.
"""
import json, os, pathlib, random, subprocess, sys
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import numpy as np, pandas as pd
from huggingface_hub import hf_hub_download

ROOT = pathlib.Path("/workspace/data/yam7h")
N_DEEP, N_VIDEO = 25, 8
rng = random.Random(0)
fail = []

for name in ("teleop", "ego"):
    out = ROOT / name
    info = json.loads((out / "meta" / "info.json").read_text())
    src = json.loads((out / "source_episodes.json").read_text())
    repo, orig_eps = src["source_repo"], src["episodes"]
    eps = [json.loads(l) for l in (out/"meta"/"episodes.jsonl").read_text().splitlines() if l.strip()]
    tasks = [json.loads(l) for l in (out/"meta"/"tasks.jsonl").read_text().splitlines() if l.strip()]
    valid_tasks = {t["task_index"] for t in tasks}
    cs = info["chunks_size"]
    print(f"\n=== {name} ({repo}) ===")
    print(f"  {info['total_episodes']:,} eps  {info['total_frames']:,} frames  "
          f"{info['total_frames']/30/3600:.3f} h   tasks.jsonl={len(tasks)}")

    # ---- payload fidelity vs the source parquet
    picks = rng.sample(range(len(orig_eps)), min(N_DEEP, len(orig_eps)))
    cols = ["observation.state", "action", "timestamp", "frame_index", "task_index"]
    bad = 0
    for new in picks:
        orig = orig_eps[new]
        dst = pd.read_parquet(out / info["data_path"].format(
            episode_chunk=new//cs, episode_index=new))
        s = pd.read_parquet(hf_hub_download(repo, info["data_path"].format(
            episode_chunk=orig//cs, episode_index=orig), repo_type="dataset"))
        for c in cols:
            if c not in dst.columns: continue
            a, b = dst[c].to_numpy(), s[c].to_numpy()
            if c in ("observation.state", "action"):
                a, b = np.stack(a), np.stack(b)
            if not np.array_equal(a, b):
                fail.append(f"{name} ep{new}(src {orig}): column {c} differs"); bad += 1
        if (dst["episode_index"] != new).any():
            fail.append(f"{name} ep{new}: episode_index not rewritten")
        if not set(dst["task_index"].unique()) <= valid_tasks:
            fail.append(f"{name} ep{new}: task_index outside tasks.jsonl")
    print(f"  payload: {len(picks)} random episodes compared to source -> "
          f"{'ALL IDENTICAL' if bad==0 else str(bad)+' MISMATCH'} "
          f"(state/action/timestamp/frame_index/task_index)")

    # ---- videos actually decode, with the right frame count
    vk = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    vbad = 0
    for new in rng.sample(range(len(eps)), min(N_VIDEO, len(eps))):
        want = eps[new]["length"]
        for key in vk:
            p = out / info["video_path"].format(
                episode_chunk=new//cs, video_key=key, episode_index=new)
            r = subprocess.run(["ffprobe","-v","error","-count_frames","-select_streams","v:0",
                                "-show_entries","stream=nb_read_frames","-of","csv=p=0",str(p)],
                               capture_output=True, text=True)
            got = r.stdout.strip()
            if not got.isdigit() or int(got) != want:
                fail.append(f"{name} ep{new} {key}: {got} video frames vs {want} rows"); vbad += 1
    print(f"  video  : {N_VIDEO} episodes x {len(vk)} cams decoded -> "
          f"{'all frame counts match rows' if vbad==0 else str(vbad)+' MISMATCH'}")

    # ---- holdout must not have leaked into teleop
    if name == "teleop":
        held = set(json.loads(pathlib.Path("/workspace/mix/teleop_holdout.json").read_text())["episodes"])
        leak = held & set(orig_eps)
        print(f"  holdout: {len(held)} eval episodes, overlap with training selection = {len(leak)}")
        if leak: fail.append(f"HOLDOUT LEAK: {sorted(leak)[:10]}")

    # ---- ego: min-eps-per-object was applied to the source pool
    if name == "ego":
        print(f"  objects: {len(src['objects_kept'])} kept at min "
              f"{src['min_eps_per_object']} eps/object (applied to source pool)")

print("\n" + "="*64)
print("AUDIT PASSED" if not fail else f"AUDIT FAILED ({len(fail)})")
for f in fail[:15]: print("  -", f)
sys.exit(1 if fail else 0)

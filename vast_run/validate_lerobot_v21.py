"""Structural validator for a LeRobot v2.1 dataset produced by mcap_to_lerobot.py.

Checks, per episode, the things that silently break training rather than erroring:

  * parquet row count == the length declared in meta/episodes.jsonl
  * timestamp == frame_index / fps  (LeRobotDataset enforces this to 1e-4 s and
    refuses the dataset otherwise)
  * frame_index 0..n-1, episode_index constant, `index` globally contiguous across
    episodes in order -- an off-by-one here misaligns every sample after it
  * every video decodes, has exactly n frames at the declared resolution and fps
  * first frame is not flat (catches an all-black or failed decode)
  * per-episode stats round-trip against the parquet, and image stats are (3,1,1)
    in [0,1] as LeRobot expects
  * total frames match meta/info.json

Usage:  python vast_run/validate_lerobot_v21.py /path/to/dataset
Exits after printing one line per episode plus a final verdict.
"""

import sys, json
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, av

root = Path(sys.argv[1])
info = json.loads((root/"meta"/"info.json").read_text())
eps  = [json.loads(l) for l in open(root/"meta"/"episodes.jsonl")]
stats= [json.loads(l) for l in open(root/"meta"/"episodes_stats.jsonl")]
tasks= [json.loads(l) for l in open(root/"meta"/"tasks.jsonl")]
src  = [json.loads(l) for l in open(root/"meta"/"source_episodes.jsonl")]
print("info:", {k:v for k,v in info.items() if k!="features"})
print("features:", list(info["features"]))
print("tasks:", tasks)

fps = info["fps"]; H,W = info["features"]["observation.images.top"]["shape"][:2]
ok = True
run = 0
for e in eps:
    i, n = e["episode_index"], e["length"]
    ch = i // info["chunks_size"]
    p = root/"data"/f"chunk-{ch:03d}"/f"episode_{i:06d}.parquet"
    t = pq.read_table(p)
    d = t.to_pydict()
    st = np.array(d["observation.state"], dtype=np.float32)
    ac = np.array(d["action"], dtype=np.float32)
    ts = np.array(d["timestamp"], dtype=np.float32)
    fi = np.array(d["frame_index"]); ei = np.array(d["episode_index"]); ix = np.array(d["index"])
    prob = []
    if t.num_rows != n: prob.append(f"rows {t.num_rows} != len {n}")
    if st.shape != (n,14): prob.append(f"state {st.shape}")
    if ac.shape != (n,14): prob.append(f"action {ac.shape}")
    if not np.allclose(ts, np.arange(n)/fps, atol=1e-6): prob.append("timestamp != i/fps")
    if not (fi == np.arange(n)).all(): prob.append("frame_index wrong")
    if not (ei == i).all(): prob.append("episode_index wrong")
    if not (ix == np.arange(run, run+n)).all(): prob.append(f"index wrong (expected start {run})")
    if not np.isfinite(st).all() or not np.isfinite(ac).all(): prob.append("non-finite proprio")
    run += n
    srcinfo = next(s for s in src if s["episode_index"]==i)
    for key in ["top","left_wrist","right_wrist"]:
        v = root/"videos"/f"chunk-{ch:03d}"/f"observation.images.{key}"/f"episode_{i:06d}.mp4"
        if not v.exists(): prob.append(f"missing {key}.mp4"); continue
        c = av.open(str(v)); vs = c.streams.video[0]
        cnt = 0; first = None; last = None
        for fr in c.decode(video=0):
            if first is None: first = fr.to_ndarray(format="rgb24")
            last = fr; cnt += 1
        lastarr = last.to_ndarray(format="rgb24") if last is not None else None
        if (vs.codec_context.height, vs.codec_context.width) != (H,W):
            prob.append(f"{key} res {vs.codec_context.height}x{vs.codec_context.width}")
        if cnt != n: prob.append(f"{key} frames {cnt} != {n}")
        if float(vs.average_rate) != fps: prob.append(f"{key} rate {vs.average_rate}")
        if first is not None and first.std() < 1.0: prob.append(f"{key} first frame ~flat (std={first.std():.2f})")
        c.close()
    # stats sanity
    ss = next(s for s in stats if s["episode_index"]==i)["stats"]
    for k in ["observation.state","action"]:
        if not np.allclose(ss[k]["mean"], (st if k=="observation.state" else ac).mean(0), atol=1e-4):
            prob.append(f"stats mean mismatch {k}")
    for key in ["top","left_wrist","right_wrist"]:
        m = np.array(ss[f"observation.images.{key}"]["mean"])
        if m.shape != (3,1,1): prob.append(f"img stats shape {m.shape}")
        if not ((m>=0).all() and (m<=1).all()): prob.append(f"img stats range {m.ravel()}")
    tag = "OK " if not prob else "BAD"
    if prob: ok = False
    print(f"{tag} ep{i:06d} n={n} {srcinfo['variant']:11s} src={srcinfo['src_size']} dur={n/fps:.1f}s "
          f"state[min={st.min():.2f} max={st.max():.2f}] " + ("| " + "; ".join(prob) if prob else ""))
print("\nTOTAL frames check:", run, "vs info", info["total_frames"], "->", run==info["total_frames"])
print("stats.json keys:", list(json.loads((root/"meta"/"stats.json").read_text())))
print("\nRESULT:", "ALL GOOD" if ok and run==info["total_frames"] else "PROBLEMS FOUND")

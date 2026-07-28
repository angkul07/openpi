#!/usr/bin/env python3
"""
MCAP -> LeRobot v2.1 converter for the abc-ego bimanual YAM dataset.

The source recordings come from two different rigs:

  variant A ("top" rig)         3 cams: /top-camera /left-wrist-camera /right-wrist-camera
                                h264, everything logged at ~30 Hz, 848x480 (a few 640x480)
  variant B ("top stereo" rig)  4 cams: /top-left-camera /top-right-camera
                                        /left-wrist-camera /right-wrist-camera
                                h265, cams ~30 Hz @ 1920x1200,
                                state ~268 Hz, action ~200 Hz

A LeRobot dataset needs one fixed schema, so both are unified onto three camera
keys -- top / left_wrist / right_wrist (variant B's /top-left-camera becomes
`top`, /top-right-camera is dropped) -- at one resolution, and every stream is
resampled onto a uniform `--fps` grid.

Layout produced (codebase_version v2.1):

    meta/info.json  meta/tasks.jsonl  meta/episodes.jsonl
    meta/episodes_stats.jsonl  meta/stats.json
    data/chunk-000/episode_000000.parquet
    videos/chunk-000/observation.images.top/episode_000000.mp4
    ...

Conversion runs in two phases:

  1. convert  (parallel, one process per episode) -- writes uuid-keyed results
     into <out>/.staging/<uuid>/.  Nothing here depends on a global episode
     index, so the phase is resumable and safe to re-run as more episodes
     finish downloading.
  2. assemble (serial, cheap) -- sorts the finished uuids, assigns contiguous
     episode indices, stamps them into the parquet files, moves the videos into
     place and writes meta/.

Run phase 1 as often as you like; run phase 2 once the source set is final.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------

CODEBASE_VERSION = "v2.1"
ROBOT_TYPE = "yam"
CHUNKS_SIZE = 1000

# camera key -> candidate source topics, in priority order.
# The stereo rig's two top cameras are the same view up to the stereo baseline
# (verified side by side), so either half works; /top-right-camera is the one we
# keep.  Everything else has a single unambiguous source.
CAMERA_TOPICS = {
    "top": ["/top-camera", "/top-right-camera"],
    "left_wrist": ["/left-wrist-camera"],
    "right_wrist": ["/right-wrist-camera"],
}
CAMERA_KEYS = list(CAMERA_TOPICS)

# 14-D vector := [left arm j0..j5, left gripper, right arm j0..j5, right gripper]
STATE_TOPICS = ["/left-arm-state", "/left-ee-state", "/right-arm-state", "/right-ee-state"]
ACTION_TOPICS = ["/left-arm-action", "/left-ee-action", "/right-arm-action", "/right-ee-action"]
MOTOR_NAMES = (
    [f"left_joint_{i}" for i in range(6)]
    + ["left_gripper"]
    + [f"right_joint_{i}" for i in range(6)]
    + ["right_gripper"]
)
STATE_DIM = len(MOTOR_NAMES)

DEFAULT_TASK = "put the screwdriver in the bin"

# foxglove CompressedVideo.format -> (raw-bitstream demuxer name, file suffix)
CODECS = {"h264": ("h264", "h264"), "h265": ("hevc", "h265"), "hevc": ("hevc", "h265")}


# --------------------------------------------------------------------------
# phase 1: per-episode conversion
# --------------------------------------------------------------------------


def _read_mcap(ep_dir: Path, tmp_dir: Path, cam_topics: dict[str, str]):
    """Single streaming pass over episode.mcap.

    Proprio is decoded into arrays; camera payloads are appended to raw
    Annex-B side files so we never hold a whole video in RAM (a 1920x1200
    episode would otherwise want >10 GB of decoded frames per camera).
    """
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    topic_by_cam = {v: k for k, v in cam_topics.items()}
    handles = {k: open(tmp_dir / f"{k}.bin", "wb") for k in cam_topics}
    cam_ts = {k: [] for k in cam_topics}
    prop_ts = {t: [] for t in STATE_TOPICS + ACTION_TOPICS}
    prop_val = {t: [] for t in STATE_TOPICS + ACTION_TOPICS}
    info = {"instruction": None, "cam_size": {}, "codec": {}}

    wanted = list(cam_topics.values()) + STATE_TOPICS + ACTION_TOPICS + ["/instruction"]
    wanted += [f"{t}-info" for t in cam_topics.values()]

    try:
        with open(ep_dir / "episode.mcap", "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for _schema, channel, message, dec in reader.iter_decoded_messages(topics=wanted):
                topic = channel.topic
                if topic in topic_by_cam:
                    cam = topic_by_cam[topic]
                    handles[cam].write(dec.data)
                    cam_ts[cam].append(message.log_time)
                    info["codec"].setdefault(cam, dec.format)
                elif topic in prop_ts:
                    prop_ts[topic].append(message.log_time)
                    prop_val[topic].append(list(dec.position))
                elif topic == "/instruction":
                    info["instruction"] = dec.data
                elif topic.endswith("-info"):
                    cam = topic_by_cam.get(topic[: -len("-info")])
                    if cam is not None:
                        info["cam_size"][cam] = (dec.width, dec.height)
    finally:
        for h in handles.values():
            h.close()

    for k in cam_ts:
        cam_ts[k] = np.asarray(cam_ts[k], dtype=np.int64)
    for t in prop_ts:
        prop_ts[t] = np.asarray(prop_ts[t], dtype=np.int64)
        prop_val[t] = np.asarray(prop_val[t], dtype=np.float64)
    return cam_ts, prop_ts, prop_val, info


def _nearest(src_ts: np.ndarray, grid_ts: np.ndarray) -> np.ndarray:
    """Index of the nearest src_ts entry for every grid_ts entry."""
    idx = np.searchsorted(src_ts, grid_ts)
    idx = np.clip(idx, 1, len(src_ts) - 1)
    left, right = src_ts[idx - 1], src_ts[idx]
    return np.where(grid_ts - left <= right - grid_ts, idx - 1, idx)


def _assemble_vec(topics, prop_ts, prop_val, grid_ts) -> np.ndarray:
    """Resample the 4 proprio streams onto the grid and concatenate to 14-D."""
    cols = []
    for t in topics:
        pick = _nearest(prop_ts[t], grid_ts)
        cols.append(prop_val[t][pick])
    out = np.concatenate(cols, axis=1)
    if out.shape[1] != STATE_DIM:
        raise ValueError(f"expected {STATE_DIM}-D vector, got {out.shape[1]} from {topics}")
    return out.astype(np.float32)


def _transcode(
    raw_path: Path,
    out_path: Path,
    demuxer: str,
    grid_to_frame: np.ndarray,
    width: int,
    height: int,
    fps: int,
    crf: int,
    preset: str,
    gop: int,
    stats_at: np.ndarray,
    fit: str = "pad",
):
    """Decode the raw bitstream, re-time onto the grid, scale, re-encode.

    Streams frame by frame so at most a couple of frames are alive at once.
    Output is always h264/yuv420p regardless of the source codec.
    """
    import av

    n = len(grid_to_frame)
    stats_set = set(int(i) for i in stats_at)
    samples = []

    tb = Fraction(1, fps)
    out = av.open(str(out_path), mode="w")
    stream = out.add_stream("libx264", rate=fps)
    stream.width, stream.height = width, height
    stream.pix_fmt = "yuv420p"
    stream.codec_context.thread_count = 1
    stream.codec_context.time_base = tb
    stream.codec_context.options = {
        "crf": str(crf),
        "preset": preset,
        "g": str(gop),
        "sc_threshold": "0",
    }

    written = 0
    src_frame = 0  # index of the decoded frame we are holding
    last_out = None
    dec_total = 0
    graph = None  # lazily built once the source geometry is known

    inp = av.open(str(raw_path), format=demuxer)
    inp.streams.video[0].thread_count = 1
    try:
        for frame in inp.decode(video=0):
            dec_total += 1
            reused = None
            while written < n and grid_to_frame[written] <= src_frame:
                if reused is None:
                    if fit == "pad":
                        if graph is None:
                            graph = _pad_graph(frame, width, height, fps)
                        graph.push(frame)
                        reused = graph.pull()
                    else:
                        # bilinear rather than swscale's default bicubic: ~2x cheaper
                        # and indistinguishable after the downscale we are doing
                        reused = frame.reformat(
                            width=width, height=height, format="yuv420p", interpolation="BILINEAR"
                        )
                    last_out = reused
                if written in stats_set:
                    # convert from the already-downscaled frame, not the source,
                    # and fold into running accumulators so we never hold frames
                    px = reused.to_ndarray(format="rgb24")[::2, ::2].reshape(-1, 3)
                    px = px.astype(np.float64) / 255.0
                    samples.append(
                        (len(px), px.sum(0), (px * px).sum(0), px.min(0), px.max(0))
                    )
                reused.pts = written
                reused.time_base = tb
                for pkt in stream.encode(reused):
                    out.mux(pkt)
                written += 1
            src_frame += 1
            if written >= n:
                break
    finally:
        inp.close()

    # decoder ran dry before the grid did -> hold the last frame
    padded = 0
    while written < n and last_out is not None:
        last_out.pts = written
        last_out.time_base = tb
        for pkt in stream.encode(last_out):
            out.mux(pkt)
        written += 1
        padded += 1

    for pkt in stream.encode():
        out.mux(pkt)
    out.close()

    if written != n:
        raise RuntimeError(f"{out_path.name}: wrote {written}/{n} frames (decoded {dec_total})")

    if samples:
        npx = sum(s[0] for s in samples)
        tot = np.sum([s[1] for s in samples], axis=0)
        totsq = np.sum([s[2] for s in samples], axis=0)
        mean = tot / npx
        var = np.maximum(totsq / npx - mean * mean, 0.0)
        stats = {
            "min": np.min([s[3] for s in samples], axis=0).reshape(3, 1, 1).tolist(),
            "max": np.max([s[4] for s in samples], axis=0).reshape(3, 1, 1).tolist(),
            "mean": mean.reshape(3, 1, 1).tolist(),
            "std": np.sqrt(var).reshape(3, 1, 1).tolist(),
            "count": [len(samples)],
        }
    else:
        z = np.zeros((3, 1, 1)).tolist()
        stats = {"min": z, "max": z, "mean": z, "std": z, "count": [0]}
    return stats, dec_total, padded


def _pad_graph(frame, width: int, height: int, fps: int):
    """scale-to-fit + centre-pad, preserving aspect ratio.

    The three rigs record at three different aspects (16:9, 16:10, 4:3).
    Stretching them all into one canvas would distort each rig by a different
    amount -- 10% for the 1920x1200 rig, 33% for the 640x480 one -- which is
    both a geometric inconsistency and a give-away cue for which rig an episode
    came from.  Padding keeps every rig's geometry true, and matches what openpi
    does downstream (`resize_with_pad` to 224x224).
    """
    import av

    ratio = max(frame.width / width, frame.height / height)
    iw = int(frame.width / ratio) & ~1  # even dims: yuv420p has half-res chroma
    ih = int(frame.height / ratio) & ~1
    graph = av.filter.Graph()
    buf = graph.add_buffer(
        width=frame.width, height=frame.height, format=frame.format.name, time_base=Fraction(1, fps)
    )
    scale = graph.add("scale", f"{iw}:{ih}:flags=bilinear")
    pad = graph.add("pad", f"{width}:{height}:{(width - iw) // 2}:{(height - ih) // 2}:black")
    fmt = graph.add("format", "yuv420p")
    sink = graph.add("buffersink")
    buf.link_to(scale)
    scale.link_to(pad)
    pad.link_to(fmt)
    fmt.link_to(sink)
    graph.configure()
    return graph


def _vec_stats(a: np.ndarray) -> dict:
    return {
        "min": a.min(0).tolist(),
        "max": a.max(0).tolist(),
        "mean": a.mean(0).tolist(),
        "std": a.std(0).tolist(),
        "count": [len(a)],
    }


def convert_episode(job) -> dict:
    ep_dir = Path(job["ep_dir"])
    uuid = ep_dir.name.removeprefix("episode_")
    stage = Path(job["staging"]) / uuid
    cfg = job["cfg"]
    t0 = time.time()

    if (stage / "meta.json").exists() and not cfg["force"]:
        return {"uuid": uuid, "status": "skipped"}

    tmp = stage.with_name(stage.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)

    try:
        from mcap.reader import make_reader

        with open(ep_dir / "episode.mcap", "rb") as f:
            present = {c.topic for c in make_reader(f).get_summary().channels.values()}

        n_src_cams = len([t for t in present if t.endswith("-camera")])
        cam_topics = {}
        for key, candidates in CAMERA_TOPICS.items():
            for cand in candidates:
                if cand in present:
                    cam_topics[key] = cand
                    break
        missing = [k for k in CAMERA_KEYS if k not in cam_topics]
        if missing:
            raise ValueError(f"no source topic for camera(s) {missing}; topics={sorted(present)}")
        for t in STATE_TOPICS + ACTION_TOPICS:
            if t not in present:
                raise ValueError(f"missing proprio topic {t}")

        cam_ts, prop_ts, prop_val, info = _read_mcap(ep_dir, tmp, cam_topics)

        for k, ts in cam_ts.items():
            if len(ts) < 2:
                raise ValueError(f"camera {k} has {len(ts)} packets")
        for t, ts in prop_ts.items():
            if len(ts) < 2:
                raise ValueError(f"{t} has {len(ts)} samples")

        # uniform grid over the window where every stream is live
        fps = cfg["fps"]
        start = max([ts[0] for ts in cam_ts.values()] + [ts[0] for ts in prop_ts.values()])
        end = min([ts[-1] for ts in cam_ts.values()] + [ts[-1] for ts in prop_ts.values()])
        n = int((end - start) * fps // 1_000_000_000) + 1
        if n < cfg["min_frames"]:
            raise ValueError(f"only {n} frames in the common window ({(end-start)/1e9:.2f}s)")
        grid_ts = start + (np.arange(n, dtype=np.int64) * 1_000_000_000) // fps

        state = _assemble_vec(STATE_TOPICS, prop_ts, prop_val, grid_ts)
        action = _assemble_vec(ACTION_TOPICS, prop_ts, prop_val, grid_ts)

        n_stats = min(cfg["stats_frames"], n)
        stats_at = np.unique(np.linspace(0, n - 1, n_stats).astype(int))

        ep_stats = {"observation.state": _vec_stats(state), "action": _vec_stats(action)}
        video_report = {}
        for key in CAMERA_KEYS:
            src_codec = (info["codec"].get(key) or "").lower()
            if src_codec not in CODECS:
                raise ValueError(f"camera {key}: unsupported codec {src_codec!r}")
            demuxer = CODECS[src_codec][0]
            g2f = _nearest(cam_ts[key], grid_ts)
            vstats, decoded, padded = _transcode(
                tmp / f"{key}.bin",
                tmp / f"{key}.mp4",
                demuxer,
                g2f,
                cfg["width"],
                cfg["height"],
                fps,
                cfg["crf"],
                cfg["preset"],
                cfg["gop"],
                stats_at,
                cfg["fit"],
            )
            ep_stats[f"observation.images.{key}"] = vstats
            video_report[key] = {
                "src_topic": cam_topics[key],
                "src_codec": src_codec,
                "src_size": list(info["cam_size"].get(key, ())),
                "src_packets": int(len(cam_ts[key])),
                "decoded_frames": decoded,
                "padded_frames": padded,
            }
            (tmp / f"{key}.bin").unlink()

        # proprio-only parquet; episode_index / index are stamped at assemble time
        pq.write_table(
            pa.table(
                {
                    "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32())),
                    "action": pa.array(action.tolist(), type=pa.list_(pa.float32())),
                    "timestamp": pa.array(np.arange(n, dtype=np.float32) / fps, type=pa.float32()),
                    "frame_index": pa.array(np.arange(n, dtype=np.int64), type=pa.int64()),
                }
            ),
            tmp / "proprio.parquet",
            compression="zstd",
        )

        subtasks = _read_annotations(ep_dir / "annotation.mcap", start, fps)

        meta = {
            "uuid": uuid,
            "source": str(ep_dir),
            "length": int(n),
            "fps": fps,
            "task": info["instruction"] or cfg["default_task"],
            # derived from what the recording actually contains, not from which
            # topic we happened to pick for `top`
            "variant": "stereo_top" if n_src_cams == 4 else "single_top",
            "n_source_cameras": n_src_cams,
            "start_ns": int(start),
            "end_ns": int(end),
            "duration_s": round((end - start) / 1e9, 3),
            "videos": video_report,
            "proprio_rates_hz": {
                t: round(float(1e9 / np.median(np.diff(ts)[np.diff(ts) > 0])), 2)
                for t, ts in prop_ts.items()
            },
            "subtasks": subtasks,
            "stats": ep_stats,
            "convert_s": round(time.time() - t0, 2),
        }
        (tmp / "meta.json").write_text(json.dumps(meta))

        shutil.rmtree(stage, ignore_errors=True)
        tmp.rename(stage)
        return {"uuid": uuid, "status": "ok", "length": int(n), "secs": meta["convert_s"]}

    except Exception as e:
        shutil.rmtree(tmp, ignore_errors=True)
        return {
            "uuid": uuid,
            "status": "failed",
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(limit=6),
        }


def _read_annotations(path: Path, start_ns: int, fps: int) -> list:
    """Sub-task annotations, kept alongside the episode (only ~10% have them)."""
    if not path.exists():
        return []
    try:
        from mcap.reader import make_reader
        from mcap_protobuf.decoder import DecoderFactory

        out = []
        with open(path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for _s, _c, message, dec in reader.iter_decoded_messages():
                out.append(
                    {
                        "frame_index": int(round((message.log_time - start_ns) / 1e9 * fps)),
                        "text": dec.data,
                    }
                )
        return out
    except Exception:
        return []


# --------------------------------------------------------------------------
# phase 2: assemble
# --------------------------------------------------------------------------


def _aggregate(per_ep: list[dict]) -> dict:
    """Count-weighted roll-up of per-episode stats into meta/stats.json."""
    out = {}
    for key in per_ep[0]:
        mins, maxs, means, varis, counts = [], [], [], [], []
        for st in per_ep:
            s = st[key]
            c = s["count"][0]
            if c == 0:
                continue
            mins.append(np.asarray(s["min"], dtype=np.float64))
            maxs.append(np.asarray(s["max"], dtype=np.float64))
            means.append(np.asarray(s["mean"], dtype=np.float64))
            varis.append(np.asarray(s["std"], dtype=np.float64) ** 2)
            counts.append(c)
        if not counts:
            continue
        w = np.asarray(counts, dtype=np.float64)
        total = w.sum()
        shape = means[0].shape
        wr = w.reshape((-1,) + (1,) * len(shape))
        mean = (np.stack(means) * wr).sum(0) / total
        var = (wr * (np.stack(varis) + (np.stack(means) - mean) ** 2)).sum(0) / total
        out[key] = {
            "min": np.min(np.stack(mins), 0).tolist(),
            "max": np.max(np.stack(maxs), 0).tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(var).tolist(),
            "count": [int(total)],
        }
    return out


def assemble(out_root: Path, cfg: dict):
    staging = out_root / ".staging"
    uuids = sorted(p.name for p in staging.iterdir() if (p / "meta.json").exists())
    if not uuids:
        sys.exit("nothing staged -- run the convert phase first")
    print(f"assembling {len(uuids)} episodes")

    for sub in ("data", "videos", "meta"):
        shutil.rmtree(out_root / sub, ignore_errors=True)
    (out_root / "meta").mkdir(parents=True)

    metas = [json.loads((staging / u / "meta.json").read_text()) for u in uuids]

    tasks, task_index = [], {}
    for m in metas:
        if m["task"] not in task_index:
            task_index[m["task"]] = len(tasks)
            tasks.append(m["task"])

    episodes, ep_stats_rows, subtask_rows = [], [], []
    running = 0
    for ep_idx, (u, m) in enumerate(zip(uuids, metas)):
        chunk = ep_idx // CHUNKS_SIZE
        n = m["length"]

        data_dir = out_root / "data" / f"chunk-{chunk:03d}"
        data_dir.mkdir(parents=True, exist_ok=True)
        t = pq.read_table(staging / u / "proprio.parquet")
        t = t.append_column("episode_index", pa.array(np.full(n, ep_idx, dtype=np.int64)))
        t = t.append_column("index", pa.array(np.arange(running, running + n, dtype=np.int64)))
        t = t.append_column(
            "task_index", pa.array(np.full(n, task_index[m["task"]], dtype=np.int64))
        )
        t = t.select(
            [
                "observation.state",
                "action",
                "timestamp",
                "frame_index",
                "episode_index",
                "index",
                "task_index",
            ]
        )
        pq.write_table(t, data_dir / f"episode_{ep_idx:06d}.parquet", compression="zstd")
        running += n

        for key in CAMERA_KEYS:
            vd = out_root / "videos" / f"chunk-{chunk:03d}" / f"observation.images.{key}"
            vd.mkdir(parents=True, exist_ok=True)
            dst = vd / f"episode_{ep_idx:06d}.mp4"
            # hardlink: staging is on the same fs and the mp4s are never mutated,
            # so this keeps .staging re-assemblable without a second copy on disk
            try:
                os.link(staging / u / f"{key}.mp4", dst)
            except OSError:
                shutil.copy2(staging / u / f"{key}.mp4", dst)

        episodes.append({"episode_index": ep_idx, "tasks": [m["task"]], "length": n})
        ep_stats_rows.append({"episode_index": ep_idx, "stats": m["stats"]})
        if m["subtasks"]:
            subtask_rows.append(
                {"episode_index": ep_idx, "uuid": u, "subtasks": m["subtasks"]}
            )

    total_frames = running
    n_eps = len(uuids)
    video_info = {
        "video.fps": float(cfg["fps"]),
        "video.height": cfg["height"],
        "video.width": cfg["width"],
        "video.channels": 3,
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "has_audio": False,
    }
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [STATE_DIM],
            "names": {"motors": MOTOR_NAMES},
        },
        "action": {"dtype": "float32", "shape": [STATE_DIM], "names": {"motors": MOTOR_NAMES}},
    }
    for key in CAMERA_KEYS:
        features[f"observation.images.{key}"] = {
            "dtype": "video",
            "shape": [cfg["height"], cfg["width"], 3],
            "names": ["height", "width", "channel"],
            "info": dict(video_info),
        }
    for name in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        features[name] = {
            "dtype": "float32" if name == "timestamp" else "int64",
            "shape": [1],
            "names": None,
        }

    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": ROBOT_TYPE,
        "total_episodes": n_eps,
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": n_eps * len(CAMERA_KEYS),
        "total_chunks": (n_eps - 1) // CHUNKS_SIZE + 1,
        "chunks_size": CHUNKS_SIZE,
        "fps": cfg["fps"],
        "splits": {"train": f"0:{n_eps}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }

    meta_dir = out_root / "meta"
    (meta_dir / "info.json").write_text(json.dumps(info, indent=4))
    _write_jsonl(meta_dir / "tasks.jsonl", [{"task_index": i, "task": t} for i, t in enumerate(tasks)])
    _write_jsonl(meta_dir / "episodes.jsonl", episodes)
    _write_jsonl(meta_dir / "episodes_stats.jsonl", ep_stats_rows)
    (meta_dir / "stats.json").write_text(
        json.dumps(_aggregate([r["stats"] for r in ep_stats_rows]), indent=4)
    )
    if subtask_rows:
        _write_jsonl(meta_dir / "subtask_annotations.jsonl", subtask_rows)
    _write_jsonl(
        meta_dir / "source_episodes.jsonl",
        [
            {
                "episode_index": i,
                "uuid": u,
                "variant": m["variant"],
                "duration_s": m["duration_s"],
                "src_size": m["videos"]["top"]["src_size"],
            }
            for i, (u, m) in enumerate(zip(uuids, metas))
        ],
    )

    hrs = total_frames / cfg["fps"] / 3600
    print(f"\n{out_root}")
    print(f"  episodes {n_eps}  frames {total_frames}  ({hrs:.2f} h @ {cfg['fps']} fps)")
    print(f"  tasks    {tasks}")
    from collections import Counter

    print(f"  variants {dict(Counter(m['variant'] for m in metas))}")
    print(f"  episodes with subtask annotations: {len(subtask_rows)}")


def _write_jsonl(path: Path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# --------------------------------------------------------------------------


def cpu_budget() -> int:
    """Usable cores.

    os.cpu_count() and sched_getaffinity both report the host's core count
    inside a container, which on a quota-limited box (vast.ai and friends)
    overshoots badly -- oversubscribing there costs far more in CFS throttle
    stalls than it gains.  Read the cgroup quota when there is one.
    """
    for path, parse in (
        ("/sys/fs/cgroup/cpu.max", lambda s: s.split()),  # cgroup v2
        ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", None),    # cgroup v1
    ):
        try:
            if parse:
                quota, period = parse(Path(path).read_text().strip())
                if quota != "max":
                    return max(1, int(float(quota) / float(period)))
            else:
                quota = int(Path(path).read_text().strip())
                period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text().strip())
                if quota > 0:
                    return max(1, quota // period)
        except (OSError, ValueError):
            continue
    return os.cpu_count() or 8


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="directory containing episode_<uuid>/ dirs")
    ap.add_argument("--out", required=True, help="output LeRobot dataset root")
    ap.add_argument("--phase", choices=["convert", "assemble", "all"], default="all")
    ap.add_argument("--workers", type=int, default=0, help="0 = one per usable core")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=848)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--preset", default="veryfast")
    ap.add_argument("--gop", type=int, default=30, help="keyframe interval; small = fast random access")
    ap.add_argument(
        "--fit",
        choices=["pad", "stretch"],
        default="pad",
        help="pad = preserve aspect with black bars (default); stretch = fill, distorting each rig differently",
    )
    ap.add_argument("--stats-frames", type=int, default=20, help="frames sampled per video for stats")
    ap.add_argument("--min-frames", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="convert at most N episodes (0 = all)")
    ap.add_argument("--only", nargs="*", default=None, help="convert only these episode uuids")
    ap.add_argument("--force", action="store_true", help="re-convert already-staged episodes")
    ap.add_argument("--default-task", default=DEFAULT_TASK)
    args = ap.parse_args()
    if args.workers <= 0:
        args.workers = cpu_budget()

    src, out_root = Path(args.src), Path(args.out)
    cfg = {
        k: getattr(args, k)
        for k in ("fps", "width", "height", "crf", "preset", "gop", "fit", "stats_frames", "min_frames", "force", "default_task")
    }
    staging = out_root / ".staging"
    staging.mkdir(parents=True, exist_ok=True)

    if args.phase in ("convert", "all"):
        eps = sorted(p for p in src.glob("episode_*") if (p / "episode.mcap").exists())
        if args.only:
            keep = {u.removeprefix("episode_") for u in args.only}
            eps = [p for p in eps if p.name.removeprefix("episode_") in keep]
        if args.limit:
            eps = eps[: args.limit]
        if not args.force:
            eps = [p for p in eps if not (staging / p.name.removeprefix("episode_") / "meta.json").exists()]
        done_already = len(list(staging.glob("*/meta.json")))
        print(f"{len(eps)} episodes to convert ({done_already} already staged), {args.workers} workers")

        jobs = [{"ep_dir": str(p), "staging": str(staging), "cfg": cfg} for p in eps]
        ok = failed = 0
        frames = 0
        t0 = time.time()
        fail_log = out_root / "failures.jsonl"
        with open(fail_log, "a") as flog, ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(convert_episode, j) for j in jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                if r["status"] == "ok":
                    ok += 1
                    frames += r["length"]
                elif r["status"] == "failed":
                    failed += 1
                    flog.write(json.dumps(r) + "\n")
                    flog.flush()
                    print(f"  FAIL {r['uuid']}: {r['error']}", flush=True)
                if i % 25 == 0 or i == len(futs):
                    el = time.time() - t0
                    rate = i / el
                    eta = (len(futs) - i) / rate / 60 if rate else 0
                    print(
                        f"  [{i}/{len(futs)}] ok={ok} fail={failed} frames={frames} "
                        f"{rate:.2f} ep/s  eta {eta:.1f} min",
                        flush=True,
                    )
        print(f"convert done in {(time.time()-t0)/60:.1f} min -- ok={ok} failed={failed}")
        if failed:
            print(f"failures logged to {fail_log}")

    if args.phase in ("assemble", "all"):
        assemble(out_root, cfg)


if __name__ == "__main__":
    main()

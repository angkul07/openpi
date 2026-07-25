"""Download the two YAM source datasets into the LeRobot cache.

Both are already LeRobot v2.1, so nothing is converted -- LeRobotDataset reads
them straight from $HF_LEROBOT_HOME/<repo_id>, which is exactly where this
script puts them.

The HF hub rate-limits hard on datasets this size (~1.6M frames, ~9k mp4s), so
this walks the file list once (one API call), downloads only what is missing,
and waits out 429s instead of dying. Safe to re-run: existing files cost zero
requests.

Usage:  uv run vast_run/dl.py            # both datasets
        uv run vast_run/dl.py teleop     # just teleop
"""

import os
import sys
import time

# Avoid the xet backend (its xet-read-token endpoint triggers 429s).
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import hf_hub_download, list_repo_files  # noqa: E402

REPOS = {
    "teleop": "angkul07/abc-teleop",
    "ego": "angkul07/EgoDex-PickPlace-YAM-14dof-multiview",
}

# LeRobotDataset(repo_id) resolves to HF_LEROBOT_HOME/<repo_id> when no explicit
# root is passed, so download there and the training config needs no path wiring.
LEROBOT_HOME = os.environ.get("HF_LEROBOT_HOME", "/workspace/.hf_home/lerobot")


def is_rate_limit(e):
    m = str(e)
    return ("429" in m or "Too Many Requests" in m
            or "cannot find the requested files" in m  # LocalEntryNotFoundError from a 429 HEAD
            or "LocalEntryNotFound" in type(e).__name__)


def missing_files(repo, dest):
    # ONE cheap API call for the file list, then a purely-local existence check.
    # Files already on disk cost ZERO http requests -> preserves the quota for
    # the files we actually still need.
    out = []
    for f in list_repo_files(repo, repo_type="dataset"):
        p = os.path.join(dest, f)
        if not (os.path.exists(p) and os.path.getsize(p) > 0):
            out.append(f)
    return out


def download(repo, dest):
    print(f"\n=== {repo} -> {dest} ===", flush=True)
    while True:
        try:
            missing = missing_files(repo, dest)
        except Exception as e:
            wait = 330 if is_rate_limit(e) else 60
            print(f"list_repo_files failed ({e}); sleeping {wait}s", flush=True)
            time.sleep(wait); continue

        print(f"MISSING {len(missing)} files", flush=True)
        if not missing:
            print(f"DONE_DOWNLOAD {repo} {dest}", flush=True)
            return

        got = 0
        for f in missing:
            while True:
                try:
                    hf_hub_download(repo, f, repo_type="dataset", local_dir=dest)
                    got += 1
                    if got % 100 == 0:
                        print(f"  ...{got}/{len(missing)} this pass", flush=True)
                    break
                except Exception as e:
                    if is_rate_limit(e):
                        print(f"  429 on {f}; waiting out 5-min quota (330s)...", flush=True)
                        time.sleep(330); continue
                    print(f"  transient err on {f}: {e}; 15s backoff", flush=True)
                    time.sleep(15); continue
        # loop: recompute missing (cheap) and mop up any that slipped through


if __name__ == "__main__":
    which = sys.argv[1:] or list(REPOS)
    unknown = [w for w in which if w not in REPOS]
    if unknown:
        raise SystemExit(f"unknown source(s) {unknown}; pick from {list(REPOS)}")
    for key in which:
        repo = REPOS[key]
        download(repo, os.path.join(LEROBOT_HOME, repo))
    print("\nALL DOWNLOADS COMPLETE", flush=True)

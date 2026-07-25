"""Upload a training run's checkpoints to a private HF model repo.

Usage:
    uv run vast_run/upload_ckpt.py                          # E-A run
    uv run vast_run/upload_ckpt.py --config pi0_fast_yam_mix_eb --exp-name eb
    uv run vast_run/upload_ckpt.py --step 25000             # one checkpoint only
"""

import argparse
import pathlib

from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="pi0_fast_yam_mix_ea")
ap.add_argument("--exp-name", default=None, help="defaults to the config's short name (ea/eb/ec)")
ap.add_argument("--step", default=None, help="upload only this checkpoint step")
ap.add_argument("--repo-id", default=None, help="defaults to angkul07/<config>")
args = ap.parse_args()

exp_name = args.exp_name or args.config.replace("pi0_fast_yam_mix_", "")
repo_id = args.repo_id or f"angkul07/{args.config.replace('_', '-')}"
folder = pathlib.Path("/workspace/openpi/checkpoints") / args.config / exp_name
path_in_repo = None
if args.step:
    folder = folder / args.step
    path_in_repo = args.step
if not folder.is_dir():
    raise SystemExit(f"no such checkpoint folder: {folder}")

api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
print(f"[upload] repo ready (private): {repo_id}", flush=True)
print(f"[upload] uploading: {folder}", flush=True)

# upload_large_folder handles the multi-GB param shards with resumable chunks, but
# it uploads a whole folder tree -- fall back to upload_folder for a single step.
if path_in_repo is None and hasattr(api, "upload_large_folder"):
    api.upload_large_folder(repo_id=repo_id, repo_type="model", folder_path=str(folder))
else:
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(folder),
        path_in_repo=path_in_repo,
        commit_message=f"Upload {args.config}/{exp_name}" + (f" step {args.step}" if args.step else ""),
    )
print(f"[upload] DONE -> https://huggingface.co/{repo_id}", flush=True)

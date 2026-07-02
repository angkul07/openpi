from huggingface_hub import HfApi

repo_id = "angkul07/pi0-fast-yam"
folder = "/workspace/openpi/checkpoints/pi0_fast_yam_low_mem_finetune/yam_run"

api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
print(f"[upload] repo ready (private): {repo_id}", flush=True)
print(f"[upload] uploading full folder: {folder}", flush=True)

if hasattr(api, "upload_large_folder"):
    api.upload_large_folder(repo_id=repo_id, repo_type="model", folder_path=folder)
else:
    api.upload_folder(
        repo_id=repo_id, repo_type="model", folder_path=folder,
        commit_message="Upload yam_run full checkpoints 2000 & 2500",
    )
print(f"[upload] DONE -> https://huggingface.co/{repo_id}", flush=True)

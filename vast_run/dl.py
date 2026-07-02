import os
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id="Kavin60606/yam_pi0fast_train", repo_type="dataset", revision="main",
    local_dir="/workspace/yam_pi0fast_train", max_workers=16,
)
print("DONE_DOWNLOAD", p, flush=True)

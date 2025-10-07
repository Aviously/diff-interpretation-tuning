import os

from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_large_folder(
    folder_path="/root/Finetune-Recovery/scripts/huggingface/scaffold",
    repo_id="diff-interpretation-tuning/loras",
    repo_type="model",
    num_workers=8,
)

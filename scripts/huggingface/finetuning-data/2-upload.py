from huggingface_hub import HfApi

from finetune_recovery import utils

# You should run `hf auth login` to login to the Hugging Face CLI before running this script.
api = HfApi()
api.upload_large_folder(
    folder_path=utils.get_repo_root()
    / "scripts"
    / "huggingface"
    / "finetuning-data"
    / "hf-repo-mirror",
    repo_id="diff-interpretation-tuning/finetuning-data",
    repo_type="dataset",
    num_workers=8,
)

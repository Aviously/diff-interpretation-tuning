import pathlib

import git
from huggingface_hub import hf_hub_download


def get_repo_root() -> pathlib.Path:
    """Returns repo root (relative to this file)."""
    return pathlib.Path(
        git.Repo(
            __file__,
            search_parent_directories=True,
        ).working_dir
    )


def parse_int_or_default(s: str, default: int) -> int:
    try:
        return int(s)
    except ValueError:
        return default


def hf_file(file_path: str) -> str:
    return hf_hub_download(
        repo_id="diff-interpretation-tuning/loras", filename=file_path
    )

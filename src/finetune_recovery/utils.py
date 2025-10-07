import pathlib

import git


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

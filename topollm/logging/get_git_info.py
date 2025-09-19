"""Get the git info of the current branch and commit hash."""

import os
import pathlib

from git import Repo


def get_git_info() -> str:
    """Get the git info of the current branch and commit hash."""
    repo = Repo(
        path=pathlib.Path(os.path.realpath(filename=__file__)).parent,
        search_parent_directories=True,
    )
    branch_name = repo.active_branch.name
    commit_hex = repo.head.object.hexsha

    info: str = f"{branch_name}/{commit_hex}"

    return info

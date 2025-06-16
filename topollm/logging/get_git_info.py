# Copyright 2024-2025
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_1 (author1@example.com)
# AUTHOR_2 (author2@example.com)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#


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

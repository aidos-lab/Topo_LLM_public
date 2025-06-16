# Copyright 2024
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


"""Load pickle files from a given directory."""

import os
import pathlib
import pickle


def load_pickle_files_from_meta_path(
    meta_path: os.PathLike,
) -> list:
    """Load pickle files stored in the respective directory."""
    data = []
    chunk_list = [f"chunk_{str(i).zfill(5)}.pkl" for i in range(len(os.listdir(meta_path)))]

    for filename in chunk_list:
        if filename.endswith(
            ".pkl",
        ):
            filepath = pathlib.Path(
                meta_path,
                filename,
            )
            with pathlib.Path(filepath).open(
                mode="rb",
            ) as f:
                chunk = pickle.load(  # noqa: S301 - This is a trusted source
                    file=f,
                )
                data.append(
                    chunk,
                )

    return data

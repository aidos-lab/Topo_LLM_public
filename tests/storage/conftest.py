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


import pathlib
import pickle

import pytest


@pytest.fixture(
    scope="session",
)
def example_batch() -> dict:
    example_data_pickle_path = pathlib.Path(
        pathlib.Path(__file__).parent,
        "example_data",
        "example_data_batch.pkl",
    )

    with open(
        file=example_data_pickle_path,
        mode="rb",
    ) as file:
        example_data = pickle.load(
            file=file,
        )

        return example_data


@pytest.fixture(
    scope="session",
)
def chunk_idx() -> int:
    return 7

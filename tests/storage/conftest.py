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

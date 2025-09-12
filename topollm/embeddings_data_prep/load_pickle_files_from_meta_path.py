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

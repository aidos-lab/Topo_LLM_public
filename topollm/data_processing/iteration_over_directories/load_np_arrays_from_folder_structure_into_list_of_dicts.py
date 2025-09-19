import logging
import os
import pathlib
import pprint

import numpy as np
from tqdm import tqdm

from topollm.path_management.parse_path_info import parse_path_info_full
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def load_np_arrays_from_folder_structure_into_list_of_dicts(
    iteration_root_dir: os.PathLike,
    pattern: str = "**/*.npy",
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> list[dict]:
    """Load numpy arrays from the folder structure and organize them in a list of dictionaries."""
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{iteration_root_dir = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"{pattern = }",  # noqa: G004 - low overhead
        )

    # Iterate over the different dataset folders in the 'iteration_root_dir' directory
    rootdir: pathlib.Path = pathlib.Path(
        iteration_root_dir,
    )
    # Only match the files compatible with the pattern
    file_path_list: list[pathlib.Path] = [
        f
        for f in rootdir.resolve().glob(
            pattern=pattern,
        )
        if f.is_file()
    ]

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{len(file_path_list) = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"file_list:\n{pprint.pformat(object=file_path_list)}",  # noqa: G004 - low overhead
        )

    loaded_data_list: list[dict] = []

    for file_path in tqdm(
        iterable=file_path_list,
        desc="Loading data from files",
    ):
        file_data = np.load(
            file=file_path,
        )

        path_info: dict = parse_path_info_full(
            path=file_path,
        )

        # Combine the path information with the flattened file data
        combined_data: dict = {
            **path_info,
            "file_path": str(object=file_path),
            "file_data": file_data,
        }

        loaded_data_list.append(
            combined_data,
        )

    return loaded_data_list

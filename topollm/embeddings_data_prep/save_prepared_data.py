"""Saving and loading of prepared data."""

import logging
import pathlib

import numpy as np
import pandas as pd

from topollm.embeddings_data_prep.prepared_data_containers import PreparedData
from topollm.path_management.embeddings.protocol import EmbeddingsPathManager
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def save_prepared_data(
    embeddings_path_manager: EmbeddingsPathManager,
    prepared_data: PreparedData,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Save the prepared data."""
    prepared_data_dir_absolute_path: pathlib.Path = embeddings_path_manager.prepared_data_dir_absolute_path

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "prepared_data_dir_absolute_path:%s",
            prepared_data_dir_absolute_path,
        )

    # Make sure the save directory exists
    pathlib.Path(prepared_data_dir_absolute_path).mkdir(
        parents=True,
        exist_ok=True,
    )

    # # # #
    # Save the prepared data
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Saving prepared data to {prepared_data_dir_absolute_path = } ...",  # noqa: G004 - low overhead
        )

    np.save(
        file=embeddings_path_manager.get_prepared_data_array_save_path(),
        arr=prepared_data.array,
    )

    prepared_data.meta_df.to_pickle(
        path=embeddings_path_manager.get_prepared_data_meta_save_path(),
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Saving prepared data to {prepared_data_dir_absolute_path = } DONE",  # noqa: G004 - low overhead
        )


def load_prepared_data(
    embeddings_path_manager: EmbeddingsPathManager,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> PreparedData:
    """Load the prepared data."""
    prepared_data_array_save_path: pathlib.Path = embeddings_path_manager.get_prepared_data_array_save_path()
    prepared_data_meta_save_path: pathlib.Path = embeddings_path_manager.get_prepared_data_meta_save_path()

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{prepared_data_array_save_path = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"{prepared_data_meta_save_path = }",  # noqa: G004 - low overhead
        )

    array = np.load(
        file=prepared_data_array_save_path,
    )

    meta_df = pd.read_pickle(  # noqa: S301 - we trust the data since it is our own
        filepath_or_buffer=prepared_data_meta_save_path,
    )

    prepared_data = PreparedData(
        array=array,
        meta_df=meta_df,
    )

    if verbosity >= Verbosity.NORMAL:
        prepared_data.log_info(
            logger=logger,
        )

    return prepared_data

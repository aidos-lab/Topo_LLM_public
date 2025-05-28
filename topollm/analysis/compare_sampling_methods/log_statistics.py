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


"""Functions for logging statistics of arrays and dataframes."""

import logging

import numpy as np
import pandas as pd

from topollm.analysis.compare_sampling_methods.filter_dataframe_based_on_filters_dict import (
    filter_dataframe_based_on_filters_dict,
)
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def log_unique_values(
    filtered_concatenated_df: pd.DataFrame,
    column_name: str = "data_subsampling_number_of_samples",
    sampling_seed_column_name: str = "data_subsampling_sampling_seed",
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """For every occurence of value in column_name, check how many rows are present in the filtered dataframe."""
    unique_values = filtered_concatenated_df[column_name].unique()

    for unique_value in unique_values:
        data_subsampling_number_of_samples_filters_dict = {
            column_name: unique_value,
        }

        filtered_concatenated_df_for_unique_value: pd.DataFrame = filter_dataframe_based_on_filters_dict(
            df=filtered_concatenated_df,
            filters_dict=data_subsampling_number_of_samples_filters_dict,
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"{unique_value = }: {filtered_concatenated_df_for_unique_value.shape = }",  # noqa: G004 - low overhead
            )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Unique {sampling_seed_column_name}:\n{filtered_concatenated_df[sampling_seed_column_name].unique()}",  # noqa: G004 - low overhead
        )


def log_statistics_of_array(
    array: np.ndarray,
    array_name: str = "default_array_name",
    logger: logging.Logger = default_logger,
) -> None:
    """Log some statistics of the array."""
    logger.info(
        msg=f"Statistics of array: {array_name = }",  # noqa: G004 - low overhead
    )
    logger.info(
        msg=f"Shape:\t{array.shape = }",  # noqa: G004 - low overhead
    )
    logger.info(
        msg=f"np.min:\t{np.min(array) = }",  # noqa: G004 - low overhead
    )
    logger.info(
        msg=f"np.max:\t{np.max(array) = }",  # noqa: G004 - low overhead
    )
    logger.info(
        msg=f"np.mean:\t{np.mean(array) = }",  # noqa: G004 - low overhead
    )
    logger.info(
        msg=f"np.std(ddof=0):\t{np.std(array, ddof=0) = }",  # noqa: G004 - low overhead
    )
    logger.info(
        msg=f"np.std(ddof=1):\t{np.std(array, ddof=1) = }",  # noqa: G004 - low overhead
    )

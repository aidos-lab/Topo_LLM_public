import logging
from typing import Any

import pandas as pd

from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def filter_dataframe_based_on_filters_dict(
    df: pd.DataFrame,
    filters_dict: dict[str, Any],
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> pd.DataFrame:
    """Filter a DataFrame based on key-value pairs specified in a dictionary.

    Args:
        df:
            The DataFrame to be filtered.
        filters_dict:
            A dictionary of column names and corresponding values to filter by.
        verbosity:
            The verbosity level of the function.
        logger:
            The logger to be used for logging.

    Returns:
        A filtered DataFrame with rows matching all key-value pairs.

    """
    subset_df: pd.DataFrame = df.copy()
    for column, value in filters_dict.items():
        new_subset_df = subset_df[subset_df[column] == value]

        if new_subset_df.empty and not subset_df.empty:  # noqa: SIM102 - we want this logic to be explicit
            # If this comparison filter yields and empty DataFrame, we log a warning.
            if verbosity >= Verbosity.NORMAL:
                logger.warning(
                    msg=f"No rows found for {column = } and {value = }",  # noqa: G004 - low overhead
                )

        subset_df = new_subset_df

    return subset_df

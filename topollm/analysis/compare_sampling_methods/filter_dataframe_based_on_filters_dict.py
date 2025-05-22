# Copyright 2024-2025
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Matthias Ruppik (mail@ruppik.net)
# Julius von Rohrscheidt (julius.rohrscheidt@helmholtz-muenchen.de)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

# Copyright 2024
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

"""Log information about a pandas DataFrame or an array."""

import logging
import pprint

import pandas as pd

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def log_dataframe_info(
    df: pd.DataFrame,
    df_name: str,
    max_log_rows: int = 20,
    *,
    check_for_nan: bool = True,
    logger: logging.Logger = default_logger,
) -> None:
    """Log information about a pandas DataFrame.

    Args:
    ----
        df (pd.DataFrame):
            The DataFrame to log information about.
        df_name (str):
            The name of the DataFrame.
        max_log_rows (int, optional):
            The maximum number of rows to log for the head and tail of the DataFrame.
            Defaults to 20.
        check_for_nan (bool, optional):
            Whether to check for NaN values in the DataFrame.
            Defaults to True.
        logger (logging.Logger, optional):
            The logger to log information to.
            Defaults to logging.getLogger(__name__).

    Returns:
        None

    Side effects:
        Logs information about the DataFrame to the logger.

    """
    logger.info(
        msg=f"{df_name}.shape:\n{df.shape}",  # noqa: G004 - low overhead
    )
    logger.info(
        msg=f"{df_name}.info():\n{df.info()}",  # noqa: G004 - low overhead
    )
    logger.info(
        msg=f"{df_name}.head({max_log_rows}):\n{df.head(max_log_rows).to_string()}",  # noqa: G004 - low overhead
    )
    logger.info(
        msg=f"{df_name}.tail({max_log_rows}):\n{df.tail(max_log_rows).to_string()}",  # noqa: G004 - low overhead
    )

    if check_for_nan:
        # Check if the dataframe contains NaN values
        has_nan = df.isna().any().any()
        logger.info(
            msg=f"has_nan:\n{has_nan}",  # noqa: G004 - low overhead
        )

        if has_nan:
            logger.warning(
                msg=f"{df_name}.isna().sum():\n{df.isna().sum()}",  # noqa: G004 - low overhead
            )
            logger.warning(
                msg="The dataframe contains NaN values. Please make sure that this is intended.",
            )

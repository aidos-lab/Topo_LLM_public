# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (ruppik@hhu.de)
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
import pathlib

import pandas as pd

from topollm.typing.enums import Verbosity

default_logger = logging.getLogger(__name__)


def save_aligned_df_and_statistics(
    aligned_df: pd.DataFrame,
    aligned_without_special_tokens_df: pd.DataFrame,
    analyzed_data_save_directory: pathlib.Path,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Save the aligned_df and the statistics of the aligned_df to csv files."""
    for current_df, current_df_description in [
        (
            aligned_df,
            "aligned_df",
        ),
        (
            aligned_without_special_tokens_df,
            "aligned_without_special_tokens_df",
        ),
    ]:
        # # # #
        # Save the current_df to a csv file
        current_df_save_path = pathlib.Path(
            analyzed_data_save_directory,
            f"{current_df_description}.csv",
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                f"{current_df_save_path = }",  # noqa: G004 - low overhead
            )
            logger.info(
                "Saving current_df to csv file ...",
            )
        current_df.to_csv(
            path_or_buf=current_df_save_path,
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                "Saving current_df to csv file DONE",
            )

        # # # #
        # Save the statistics of the current_df to a csv file
        current_df_statistics_save_path = pathlib.Path(
            analyzed_data_save_directory,
            f"{current_df_description}_statistics.csv",
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                f"{current_df_statistics_save_path = }",  # noqa: G004 - low overhead
            )
            logger.info(
                "Saving statistics to file ...",
            )
        current_df.describe().to_csv(
            path_or_buf=current_df_statistics_save_path,
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                "Saving statistics to file DONE",
            )

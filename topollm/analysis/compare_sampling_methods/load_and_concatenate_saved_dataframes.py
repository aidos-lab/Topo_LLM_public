# Copyright 2024-2025
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (mail@ruppik.net)
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

"""Load and concatenate saved dataframes from the specified directory."""

import logging
import pathlib
import pprint

import pandas as pd

from topollm.config_classes.constants import NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS
from topollm.logging.log_dataframe_info import log_dataframe_info
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def load_and_concatenate_saved_dataframes(
    root_dir: pathlib.Path,
    pattern: str = "full_local_estimates_df.csv",
    save_path: pathlib.Path | None = None,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> pd.DataFrame:
    """Load and concatenate saved dataframes from the specified directory."""
    # Initialize an empty list to store dataframes
    dfs = []

    # # # #
    # Traverse the directory structure using pathlib's rglob
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Iterating over {root_dir = } ...",  # noqa: G004 - low overhead
        )

    for file_path in root_dir.rglob(
        pattern=pattern,
    ):
        # Load the CSV file into a dataframe
        current_df = None

        try:
            current_df = pd.read_csv(
                filepath_or_buffer=file_path,
                keep_default_na=False,
            )
            # Append the dataframe to the list
            dfs.append(
                current_df,
            )
        except FileNotFoundError as e:
            logger.exception(
                msg=f"Error reading {file_path = }: {e}",  # noqa: G004 - low overhead
            )
            logger.warning(
                msg=f"Skipping {file_path = }",  # noqa: G004 - low overhead
            )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Iterating over {root_dir = } DONE",  # noqa: G004 - low overhead
        )

    # # # #
    # Concatenate the dataframes
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Concatenating dataframes ...",
        )

    if dfs:
        concatenated_df: pd.DataFrame = pd.concat(
            objs=dfs,
            ignore_index=True,
        )
    else:
        logger.info(
            msg=f"No files found with pattern {pattern = } in {root_dir = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg="Returning empty dataframe.",
        )
        concatenated_df = pd.DataFrame()  # Empty dataframe if no files found

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Concatenating dataframes DONE",
        )

    # NOTE: Since some of the columns contain NaN values, we cannot cast the columns to specific types here

    # # # #
    # Save the concatenated dataframe
    if save_path is not None:
        logger.info(
            msg=f"Saving concatenated dataframe to {save_path = } ...",  # noqa: G004 - low overhead
        )
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        concatenated_df.to_csv(
            path_or_buf=save_path,
            index=False,
        )
        logger.info(
            msg=f"Saving concatenated dataframe to {save_path = } DONE",  # noqa: G004 - low overhead
        )

    # # # #
    # Log information about the concatenated dataframe
    if verbosity >= Verbosity.NORMAL:
        log_dataframe_info(
            df=concatenated_df,
            df_name="concatenated_df",
            logger=logger,
        )

    columns_to_investigate: list[str] = [
        "data_full",
        "data_subsampling_full",
        "model_partial_name",
    ]

    if verbosity >= Verbosity.NORMAL:
        for column_name in columns_to_investigate:
            if column_name not in concatenated_df.columns:
                continue

            unique_values = concatenated_df[column_name].unique()
            logger.info(
                msg=f"Unique values in column '{column_name = }': {len(unique_values)}",  # noqa: G004 - low overhead
            )

            logger.info(
                msg=f"concatenated_df[{column_name}].unique():{pprint.pformat(unique_values)}",  # noqa: G004 - low overhead
            )

    return concatenated_df

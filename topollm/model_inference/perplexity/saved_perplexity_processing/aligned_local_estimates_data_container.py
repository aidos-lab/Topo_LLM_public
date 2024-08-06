# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (ruppik@hhu.de)
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

"""Container for aligned local estimates data and methods to operate on these."""

import logging
import pathlib

import pandas as pd

from topollm.config_classes.main_config import MainConfig
from topollm.model_inference.perplexity.saved_perplexity_processing.correlation_analysis import (
    compute_and_save_correlation_results_on_all_input_columns,
    extract_correlation_columns,
)
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.path_management.embeddings.protocol import EmbeddingsPathManager
from topollm.typing.enums import Verbosity

default_logger = logging.getLogger(__name__)


class AlignedLocalEstimatesDataContainer:
    """Container for aligned local estimates data and methods to operate on these."""

    def __init__(
        self,
        main_config_for_perplexity: MainConfig,
        main_config_for_local_estimates: MainConfig,
        aligned_df: pd.DataFrame,
        aligned_without_special_tokens_df: pd.DataFrame,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Initialize the container."""
        self.main_config_for_perplexity = main_config_for_perplexity
        self.main_config_for_local_estimates = main_config_for_local_estimates
        self.aligned_df = aligned_df
        self.aligned_without_special_tokens_df = aligned_without_special_tokens_df

        self.verbosity = verbosity
        self.logger = logger

        self.local_estimates_embeddings_path_manager = get_embeddings_path_manager(
            main_config=self.main_config_for_local_estimates,
            logger=logger,
        )

    @property
    def embeddings_path_manager(
        self,
    ) -> EmbeddingsPathManager:
        return self.local_estimates_embeddings_path_manager

    def save_aligned_df_and_statistics(
        self,
    ) -> None:
        """Save the aligned_df and the statistics of the aligned_df to csv files."""
        for current_df, current_df_description in [
            (
                self.aligned_df,
                "aligned_df",
            ),
            (
                self.aligned_without_special_tokens_df,
                "aligned_without_special_tokens_df",
            ),
        ]:
            # # # #
            # Save the current_df to a csv file
            current_df_save_path = self.embeddings_path_manager.get_aligned_df_save_path(
                aligned_df_file_name=f"{current_df_description}.csv",
            )
            current_df_save_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    f"{current_df_save_path = }",  # noqa: G004 - low overhead
                )
                self.logger.info(
                    "Saving current_df to csv file ...",
                )
            current_df.to_csv(
                path_or_buf=current_df_save_path,
            )
            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    "Saving current_df to csv file DONE",
                )

            # # # #
            # Save the statistics of the current_df to a csv file
            current_df_statistics_save_path = self.embeddings_path_manager.get_aligned_df_save_path(
                aligned_df_file_name=f"{current_df_description}_statistics.csv",
            )
            current_df_statistics_save_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    f"{current_df_statistics_save_path = }",  # noqa: G004 - low overhead
                )
                self.logger.info(
                    "Saving statistics to file ...",
                )
            current_df.describe().to_csv(
                path_or_buf=current_df_statistics_save_path,
            )
            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    "Saving statistics to file DONE",
                )

    def run_point_level_correlation_analysis_and_save(
        self,
        correlation_columns: list[str] | None = None,
    ) -> None:
        """Run the correlation analysis on the aligned_df and save the results."""
        only_correlation_columns_aligned_df: pd.DataFrame = extract_correlation_columns(
            aligned_df=self.aligned_df,
            correlation_columns=correlation_columns,
            verbosity=self.verbosity,
            logger=self.logger,
        )

        compute_and_save_correlation_results_on_all_input_columns(
            only_correlation_columns_aligned_df=only_correlation_columns_aligned_df,
            embeddings_path_manager=self.embeddings_path_manager,
            verbosity=self.verbosity,
            logger=self.logger,
        )

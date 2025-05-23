# Copyright 2024
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_1 (author1@example.com)
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

"""Load aligned_df.csv files from a directory and compute aggregated statistics for comparison."""

import logging
import pathlib
from typing import TYPE_CHECKING

import hydra
import hydra.core.hydra_config
import omegaconf
from tqdm import tqdm

from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.model_inference.perplexity.saved_perplexity_processing.correlation.compute_correlations_and_plots_from_aggregated_statistics import (
    compute_and_save_correlations_from_aggregated_statistics_df,
)
from topollm.model_inference.perplexity.saved_perplexity_processing.correlation.load_aligned_dfs_and_create_aggregated_statistics_and_analyse_data import (
    load_aligned_dfs_and_create_aggregated_statistics_and_analyse_data,
)
from topollm.path_management.embeddings.factory import get_embeddings_path_manager

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig
    from topollm.path_management.embeddings.protocol import EmbeddingsPathManager

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)

global_logger: logging.Logger = logging.getLogger(
    name=__name__,
)

setup_exception_logging(
    logger=global_logger,
)


@hydra.main(
    config_path=f"{HYDRA_CONFIGS_BASE_PATH}",
    config_name="main_config",
    version_base="1.3",
)
def main(
    config: omegaconf.DictConfig,
) -> None:
    """Run the script."""
    logger = global_logger
    logger.info("Running script ...")

    main_config: MainConfig = initialize_configuration(
        config=config,
        logger=logger,
    )
    verbosity = main_config.verbosity

    create_individual_dataset_analysis = True
    skip_loading_of_aligned_dfs_and_aggregating_statistics = False

    if create_individual_dataset_analysis:
        dataset_name_list = [
            None,  # 'None' means all datasets
            "iclr_2024_submissions_split-test_ctxt-dataset_entry_samples--1_feat-col-ner_tags",
            "iclr_2024_submissions_split-train_ctxt-dataset_entry_samples-5000_feat-col-ner_tags",
            "iclr_2024_submissions_split-validation_ctxt-dataset_entry_samples--1_feat-col-ner_tags",
            "multiwoz21_split-test_ctxt-dataset_entry_samples-3000_feat-col-ner_tags",
            "multiwoz21_split-train_ctxt-dataset_entry_samples-10000_feat-col-ner_tags",
            "multiwoz21_split-validation_ctxt-dataset_entry_samples-3000_feat-col-ner_tags",
            "one-year-of-tsla-on-reddit_split-test_ctxt-dataset_entry_samples-3000_feat-col-ner_tags",
            "one-year-of-tsla-on-reddit_split-train_ctxt-dataset_entry_samples-10000_feat-col-ner_tags",
            "one-year-of-tsla-on-reddit_split-validation_ctxt-dataset_entry_samples-3000_feat-col-ner_tags",
            "sgd_split-test_ctxt-dataset_entry_samples-3000_feat-col-ner_tags",
            "sgd_split-validation_ctxt-dataset_entry_samples-3000_feat-col-ner_tags",
            "wikitext_split-test_ctxt-dataset_entry_samples--1_feat-col-ner_tags",
            "wikitext_split-validation_ctxt-dataset_entry_samples--1_feat-col-ner_tags",
        ]
    else:
        dataset_name_list = [
            None,  # 'None' means all datasets
        ]

    embeddings_path_manager: EmbeddingsPathManager = get_embeddings_path_manager(
        main_config=main_config,
        logger=logger,
    )

    root_directory = pathlib.Path(
        embeddings_path_manager.data_dir,
        "analysis",
        "aligned_and_analyzed",
        "twonn",
    )
    output_directory = pathlib.Path(
        embeddings_path_manager.saved_plots_dir_absolute_path,
        "correlation_analyis",
    )

    if not skip_loading_of_aligned_dfs_and_aggregating_statistics:
        for dataset_name in tqdm(
            dataset_name_list,
            desc="Processing datasets",
        ):
            for statistic in [
                "mean",
                "std",
            ]:
                aggregated_statistics = load_aligned_dfs_and_create_aggregated_statistics_and_analyse_data(
                    root_directory=root_directory,
                    output_directory=output_directory,
                    dataset_name=dataset_name,
                    statistic=statistic,
                    verbosity=verbosity,
                    logger=logger,
                )

                if statistic == "mean":
                    compute_and_save_correlations_from_aggregated_statistics_df(
                        df=aggregated_statistics,
                        output_folder=output_directory,
                        verbosity=verbosity,
                        logger=logger,
                    )

    logger.info("Running script DONE")


if __name__ == "__main__":
    main()

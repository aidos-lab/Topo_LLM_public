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

"""Load aligned_df.csv files from a directory and compute aggregated statistics for comparison."""

import logging
import os
import pathlib
from typing import TYPE_CHECKING

import hydra
import hydra.core.hydra_config
import omegaconf
from tqdm import tqdm

from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
from topollm.config_classes.setup_OmegaConf import setup_omega_conf
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.model_inference.perplexity.saved_perplexity_processing.correlation.aligned_df_containers import (
    AlignedDF,
    AlignedDFCollection,
)
from topollm.model_inference.perplexity.saved_perplexity_processing.correlation.plot_statistics_comparison import (
    plot_statistics_comparison,
)
from topollm.model_inference.perplexity.saved_perplexity_processing.correlation_analysis import (
    compute_and_save_correlation_results_via_mapping_on_all_input_columns,
)
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig
    from topollm.path_management.embeddings.protocol import EmbeddingsPathManager

default_logger = logging.getLogger(__name__)

global_logger = logging.getLogger(__name__)

setup_exception_logging(
    logger=global_logger,
)

setup_omega_conf()


def find_aligned_dfs(
    root_dir: os.PathLike,
    dataset: str | None = None,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> AlignedDFCollection:
    """Recursively find all aligned_df.csv files in the given directory that match the dataset pattern.

    Args:
    ----
        root_dir:
            Root directory to start the search.
        dataset:
            The dataset identifier to filter the models.
        verbosity:
            The verbosity level for logging.
        logger:
            The logger instance to use for logging.

    Returns:
    -------
        AlignedDFCollection: A collection of AlignedDF objects.

    """
    aligned_df_collection = AlignedDFCollection()

    for dirpath, _, filenames in tqdm(
        os.walk(
            root_dir,
        ),
    ):
        if "aligned_df.csv" in filenames:
            file_path = pathlib.Path(
                dirpath,
                "aligned_df.csv",
            )

            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    f"Found aligned_df.csv file: {file_path = }",  # noqa: G004 - low overhead
                )

            aligned_df_object = AlignedDF(
                file_path=file_path,
            )

            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    f"{aligned_df_object.metadata = }",  # noqa: G004 - low overhead
                )

            if aligned_df_object.metadata.dataset == dataset or dataset is None:
                aligned_df_collection.add_aligned_df(
                    aligned_df=aligned_df_object,
                )

    return aligned_df_collection


def load_aligned_dfs_and_analyse_data(
    root_directory: pathlib.Path,
    output_directory: pathlib.Path,
    dataset_name: str | None = None,
    statistic: str = "mean",
    *,
    show_plot: bool = False,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Load aligned_df.csv files from a directory and compute aggregated statistics for comparison."""
    results_save_directory = pathlib.Path(
        output_directory,
        str(dataset_name),
        f"statistic-{statistic}",
    )

    aligned_df_collection: AlignedDFCollection = find_aligned_dfs(
        root_dir=root_directory,
        dataset=dataset_name,
        verbosity=verbosity,
        logger=logger,
    )
    logger.info(
        f"Found {len(aligned_df_collection) = } aligned_df.csv files.",  # noqa: G004 - low overhead
    )

    aggregated_statistics = aligned_df_collection.get_aggregated_statistics(
        statistic=statistic,
    )

    # # # #
    # Save the aggregated statistics to a CSV file
    aggregated_statistics_save_path = pathlib.Path(
        results_save_directory,
        "aggregated_statistics.csv",
    )
    aggregated_statistics_save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    aggregated_statistics.to_csv(
        aggregated_statistics_save_path,
    )

    # # # #
    # Compute and save correlation matrices
    correlation_columns = [
        "token_perplexity",
        "token_log_perplexity",
        "local_estimate",
    ]

    correlation_results_save_directory = pathlib.Path(
        results_save_directory,
        "correlation_results",
    )

    compute_and_save_correlation_results_via_mapping_on_all_input_columns(
        only_correlation_columns_df=aggregated_statistics[correlation_columns],
        output_directory=correlation_results_save_directory,
        verbosity=verbosity,
        logger=logger,
    )

    # # # #
    # Plot comparison across checkpoints
    plot_statistics_comparison(
        df=aggregated_statistics,
        output_dir=results_save_directory,
        show_plot=show_plot,
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

    dataset_name_list = [
        None,
        "iclr_2024_submissions_split-test_ctxt-dataset_entry_samples--1_feat-col-ner_tags",
        "iclr_2024_submissions_split-validation_ctxt-dataset_entry_samples--1_feat-col-ner_tags",
        "multiwoz21_split-test_ctxt-dataset_entry_samples-3000_feat-col-ner_tags",
        "multiwoz21_split-validation_ctxt-dataset_entry_samples-3000_feat-col-ner_tags",
        "one-year-of-tsla-on-reddit_split-test_ctxt-dataset_entry_samples-3000_feat-col-ner_tags",
        "one-year-of-tsla-on-reddit_split-validation_ctxt-dataset_entry_samples-3000_feat-col-ner_tags",
        "sgd_split-test_ctxt-dataset_entry_samples-3000_feat-col-ner_tags",
        "sgd_split-validation_ctxt-dataset_entry_samples-3000_feat-col-ner_tags",
        "wikitext_split-test_ctxt-dataset_entry_samples--1_feat-col-ner_tags",
        "wikitext_split-validation_ctxt-dataset_entry_samples--1_feat-col-ner_tags",
    ]

    statistic = "mean"

    for dataset_name in tqdm(
        dataset_name_list,
        desc="Processing datasets",
    ):
        load_aligned_dfs_and_analyse_data(
            root_directory=root_directory,
            output_directory=output_directory,
            dataset_name=dataset_name,
            statistic=statistic,
            verbosity=verbosity,
            logger=logger,
        )

    logger.info("Running script DONE")


if __name__ == "__main__":
    main()

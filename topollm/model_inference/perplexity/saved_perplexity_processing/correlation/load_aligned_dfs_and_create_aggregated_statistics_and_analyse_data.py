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

import logging
import pathlib

import pandas as pd

from topollm.model_inference.perplexity.saved_perplexity_processing.correlation.aligned_df_containers import (
    AlignedDFCollection,
)
from topollm.model_inference.perplexity.saved_perplexity_processing.correlation.correlation_analysis import (
    compute_and_save_correlation_results_via_mapping_on_all_input_columns,
)
from topollm.model_inference.perplexity.saved_perplexity_processing.correlation.find_aligned_dfs import find_aligned_dfs
from topollm.model_inference.perplexity.saved_perplexity_processing.plot.plot_statistics_comparison import (
    plot_statistics_comparison,
)
from topollm.typing.enums import Verbosity

default_logger = logging.getLogger(__name__)


def load_aligned_dfs_and_create_aggregated_statistics_and_analyse_data(
    root_directory: pathlib.Path,
    output_directory: pathlib.Path,
    dataset_name: str | None = None,
    statistic: str = "mean",
    *,
    show_plot: bool = False,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> pd.DataFrame:
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

    aggregated_statistics: pd.DataFrame = aligned_df_collection.get_aggregated_statistics(
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

    return aggregated_statistics

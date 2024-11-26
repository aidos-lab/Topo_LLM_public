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


"""Functions to create histograms over different subsampling number of samples for the concatenated dataframe."""

import logging
import pathlib
from itertools import product

import pandas as pd
from tqdm import tqdm

from topollm.analysis.compare_sampling_methods.filter_dataframe_based_on_filters_dict import (
    filter_dataframe_based_on_filters_dict,
)
from topollm.analysis.compare_sampling_methods.log_statistics import log_unique_values
from topollm.analysis.compare_sampling_methods.make_plots import (
    Y_AXIS_LIMITS,
    PlotSavePathCollection,
    create_boxplot_of_mean_over_different_sampling_seeds,
    generate_fixed_params_text,
)
from topollm.config_classes.constants import NAME_PREFIXES_TO_FULL_DESCRIPTIONS, TOPO_LLM_REPOSITORY_BASE_PATH
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def create_histograms_over_data_subsampling_number_of_samples(
    concatenated_df: pd.DataFrame,
    concatenated_filters_dict: dict,
    figsize: tuple[int, int] = (24, 8),
    common_prefix_path: pathlib.Path | None = None,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Create histograms over the data_subsampling_number_of_samples column."""
    filtered_concatenated_df: pd.DataFrame = filter_dataframe_based_on_filters_dict(
        df=concatenated_df,
        filters_dict=concatenated_filters_dict,
    )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{filtered_concatenated_df.shape = }",  # noqa: G004 - low overhead
        )

    log_unique_values(
        filtered_concatenated_df=filtered_concatenated_df,
        column_name="data_subsampling_number_of_samples",
        sampling_seed_column_name="data_subsampling_sampling_seed",
        verbosity=verbosity,
        logger=logger,
    )

    # # # #
    # Select the data for the analysis
    data_for_different_data_subsampling_number_of_samples_analysis_df: pd.DataFrame = filtered_concatenated_df

    fixed_params_text: str = generate_fixed_params_text(
        filters_dict=concatenated_filters_dict,
    )

    x_column_name = "data_subsampling_number_of_samples"

    for y_min, y_max in Y_AXIS_LIMITS.values():
        plot_save_path_collection: PlotSavePathCollection = PlotSavePathCollection.create_from_common_prefix_path(
            common_prefix_path=common_prefix_path,
            plot_file_name=f"{y_min=}_{y_max=}.pdf",
        )

        create_boxplot_of_mean_over_different_sampling_seeds(
            subset_local_estimates_df=data_for_different_data_subsampling_number_of_samples_analysis_df,
            plot_save_path_collection=plot_save_path_collection,
            x_column_name=x_column_name,
            y_column_name="array_data_truncated_mean",
            seed_column_name="data_subsampling_sampling_seed",
            fixed_params_text=fixed_params_text,
            figsize=figsize,
            y_min=y_min,
            y_max=y_max,
            verbosity=verbosity,
            logger=logger,
        )


def run_data_subsampling_number_of_samples_analysis(
    concatenated_df: pd.DataFrame,
    data_full_list_to_process: list[str],
    data_subsampling_split_list_to_process: list[str],
    data_subsampling_sampling_mode_list_to_process: list[str],
    model_full_list_to_process: list[str],
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Run the analysis over the different combinations of data, subsampling methods, and models."""
    product_to_process = product(
        data_full_list_to_process,
        data_subsampling_split_list_to_process,
        data_subsampling_sampling_mode_list_to_process,
        model_full_list_to_process,
    )
    product_to_process_list = list(product_to_process)

    for (
        data_full,
        data_subsampling_split,
        data_subsampling_sampling_mode,
        model_full,
    ) in tqdm(
        iterable=product_to_process_list,
        desc="Processing different combinations",
        total=len(product_to_process_list),
    ):
        concatenated_filters_dict = {
            "data_full": data_full,
            "model_full": model_full,
            "data_subsampling_split": data_subsampling_split,
            "data_subsampling_sampling_mode": data_subsampling_sampling_mode,
            "data_prep_sampling_method": "random",
            "data_prep_sampling_samples": 150_000,
            NAME_PREFIXES_TO_FULL_DESCRIPTIONS["dedup"]: "array_deduplicator",
            "local_estimates_samples": 60_000,
            "n_neighbors": 128,
        }

        common_prefix_path = pathlib.Path(
            TOPO_LLM_REPOSITORY_BASE_PATH,
            "data",
            "saved_plots",
            "mean_estimates_over_different_data_subsampling_number_of_samples",
            f"{data_full=}",
            f"{data_subsampling_split=}",
            f"{data_subsampling_sampling_mode=}",
            f"{model_full=}",
        )

        create_histograms_over_data_subsampling_number_of_samples(
            concatenated_df=concatenated_df,
            concatenated_filters_dict=concatenated_filters_dict,
            figsize=(24, 8),
            common_prefix_path=common_prefix_path,
            verbosity=verbosity,
            logger=logger,
        )

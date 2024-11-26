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

"""Functions to create histograms over the different model checkpoints for the concatenated dataframe."""

import logging
import pathlib
from itertools import product

import pandas as pd
from tqdm import tqdm

from topollm.analysis.compare_sampling_methods.checkpoint_analysis.model_loss_extractor import ModelLossExtractor
from topollm.analysis.compare_sampling_methods.filter_dataframe_based_on_filters_dict import (
    filter_dataframe_based_on_filters_dict,
)
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


def create_histograms_over_model_checkpoints(
    concatenated_df: pd.DataFrame,
    concatenated_filters_dict: dict,
    base_model_partial_name: str = "model=roberta-base",
    figsize: tuple[int, int] = (24, 8),
    common_prefix_path: pathlib.Path | None = None,
    model_loss_extractor: ModelLossExtractor | None = None,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Create histograms over the different model checkpoints for the concatenated dataframe."""
    filtered_concatenated_df: pd.DataFrame = filter_dataframe_based_on_filters_dict(
        df=concatenated_df,
        filters_dict=concatenated_filters_dict,
        verbosity=verbosity,
        logger=logger,
    )

    # # # #
    # Filter for the dataframe with just the base model data

    same_filters_but_for_base_model = concatenated_filters_dict.copy()
    same_filters_but_for_base_model["model_partial_name"] = base_model_partial_name
    # We need to drop the entry for "model_seed", because for the base model this value is empty
    same_filters_but_for_base_model.pop("model_seed")

    filtered_for_base_model_concatenated_df = filter_dataframe_based_on_filters_dict(
        df=concatenated_df,
        filters_dict=same_filters_but_for_base_model,
    )

    # Set all the values in the "model_checkpoint" column to "-1"
    filtered_for_base_model_concatenated_df[NAME_PREFIXES_TO_FULL_DESCRIPTIONS["ckpt"]] = -1

    # # # #
    # Create a dataframe by concatenating the two dataframes
    data_for_checkpoint_analysis_df: pd.DataFrame = pd.concat(
        objs=[filtered_concatenated_df, filtered_for_base_model_concatenated_df],
        ignore_index=True,
    )

    # # # #
    # Group "data_for_checkpoint_analysis_df" by value in 'model_checkpoint' column
    # and make a boxplot of "array_data_truncated_mean" for each group

    fixed_params_text: str = generate_fixed_params_text(
        filters_dict=concatenated_filters_dict,
    )

    # # # #
    # Optional: Load the model losses if available
    if model_loss_extractor:
        model_losses_df = model_loss_extractor.get_model_losses_over_finetuning_checkpoints(
            data_full=concatenated_filters_dict["data_full"],
            data_subsampling_split=concatenated_filters_dict["data_subsampling_split"],
            model_partial_name=concatenated_filters_dict["model_partial_name"],
            language_model_seed=concatenated_filters_dict["model_seed"],
        )

        if common_prefix_path is not None:
            model_losses_save_path = pathlib.Path(
                common_prefix_path,
                "raw_data",
                "model_losses.csv",
            )
            # Save the model losses to a CSV file if available
            if model_losses_df is not None:
                model_losses_save_path.parent.mkdir(parents=True, exist_ok=True)
                model_losses_df.to_csv(
                    path_or_buf=model_losses_save_path,
                    index=False,
                )
    else:
        logger.info(msg="model_loss_extractor=None, so model_losses_df=None.")
        model_losses_df = None

    # # # #
    # Create the plots
    for y_min, y_max in Y_AXIS_LIMITS.values():
        plot_save_path_collection: PlotSavePathCollection = PlotSavePathCollection.create_from_common_prefix_path(
            common_prefix_path=common_prefix_path,
            plot_file_name=f"{y_min=}_{y_max=}.pdf",
        )

        create_boxplot_of_mean_over_different_sampling_seeds(
            subset_local_estimates_df=data_for_checkpoint_analysis_df,
            plot_save_path_collection=plot_save_path_collection,
            x_column_name=NAME_PREFIXES_TO_FULL_DESCRIPTIONS["ckpt"],
            y_column_name="array_data_truncated_mean",
            fixed_params_text=fixed_params_text,
            model_losses_df=model_losses_df,
            figsize=figsize,  # This should be a bit larger than the default size, because we have many checkpoints to show
            y_min=y_min,
            y_max=y_max,
            verbosity=verbosity,
            logger=logger,
        )


def run_checkpoint_analysis_over_different_data_and_models(
    concatenated_df: pd.DataFrame,
    data_full_list_to_process: list[str],
    data_subsampling_split_list_to_process: list[str],
    data_subsampling_sampling_mode_list_to_process: list[str],
    model_partial_name_list_to_process: list[str],
    language_model_seed_list_to_process: list[int],
    model_loss_extractor: ModelLossExtractor | None = None,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Run the checkpoint analysis over different data and models."""
    product_to_process = product(
        data_full_list_to_process,
        data_subsampling_split_list_to_process,
        data_subsampling_sampling_mode_list_to_process,
        model_partial_name_list_to_process,
        language_model_seed_list_to_process,
    )
    product_to_process_list = list(product_to_process)

    for (
        data_full,
        data_subsampling_split,
        data_subsampling_sampling_mode,
        model_partial_name,
        language_model_seed,
    ) in tqdm(
        iterable=product_to_process_list,
        desc="Processing different combinations of data_full, data_subsampling_split, and model_partial_name",
        total=len(product_to_process_list),
    ):
        concatenated_filters_dict = {
            "data_full": data_full,
            "data_subsampling_split": data_subsampling_split,
            "data_subsampling_sampling_mode": data_subsampling_sampling_mode,
            "data_subsampling_number_of_samples": 10_000,
            "model_partial_name": model_partial_name,
            "model_seed": language_model_seed,
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
            "mean_estimates_over_different_checkpoints",
            f"{data_full=}",
            f"{data_subsampling_split=}",
            f"{data_subsampling_sampling_mode=}",
            f"{model_partial_name=}",
            f"{language_model_seed=}",
        )

        create_histograms_over_model_checkpoints(
            concatenated_df=concatenated_df,
            concatenated_filters_dict=concatenated_filters_dict,
            figsize=(22, 8),
            common_prefix_path=common_prefix_path,
            model_loss_extractor=model_loss_extractor,
            verbosity=verbosity,
            logger=logger,
        )

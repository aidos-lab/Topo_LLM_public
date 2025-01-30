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

"""Create plots to compare mean local estimates with mean losses for different models."""

import itertools
import logging
import pathlib
from collections.abc import Generator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import hydra
import omegaconf
import pandas as pd
from tqdm import tqdm

from topollm.analysis.correlation.compute_correlations_with_count import compute_correlations_with_count
from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
from topollm.config_classes.setup_OmegaConf import setup_omega_conf
from topollm.data_processing.iteration_over_directories.load_json_dicts_from_folder_structure_into_df import (
    load_json_dicts_from_folder_structure_into_df,
)
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.log_dataframe_info import log_dataframe_info
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.plotting.create_scatter_plot import create_scatter_plot
from topollm.plotting.line_plot_grouped_by_categorical_column import (
    line_plot_grouped_by_categorical_column,
)
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig
    from topollm.path_management.embeddings.protocol import EmbeddingsPathManager

# Logger for this file
global_logger: logging.Logger = logging.getLogger(
    name=__name__,
)
default_logger: logging.Logger = logging.getLogger(
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
    """Run main function."""
    logger: logging.Logger = global_logger
    logger.info(
        msg="Running script ...",
    )

    # ================================================== #
    # Load configuration and initialize path manager
    # ================================================== #

    main_config: MainConfig = initialize_configuration(
        config=config,
        logger=logger,
    )
    verbosity: Verbosity = main_config.verbosity

    embeddings_path_manager: EmbeddingsPathManager = get_embeddings_path_manager(
        main_config=main_config,
        logger=logger,
    )

    # ================================================== #
    # Load data and create plots
    # ================================================== #

    # The following directory contains the different dataset folders.
    # Logging of the directory is done in the function which iterates over the directories.
    iteration_root_dir = pathlib.Path(
        embeddings_path_manager.get_distances_and_influence_on_local_estimates_root_dir_absolute_path(),
        main_config.analysis.investigate_distances.get_config_description(),
        main_config.local_estimates.method_description,  # For example: 'twonn'
    )

    descriptive_statistics_df: pd.DataFrame = load_json_dicts_from_folder_structure_into_df(
        iteration_root_dir=iteration_root_dir,
        verbosity=verbosity,
        logger=logger,
    )

    # # # #
    # Filter the DataFrame for selected settings
    tokenizer_add_prefix_space = "False"

    filtered_descriptive_statistics_df: pd.DataFrame = descriptive_statistics_df.copy()
    filtered_descriptive_statistics_df = filtered_descriptive_statistics_df[
        (filtered_descriptive_statistics_df["tokenizer_add_prefix_space"] == tokenizer_add_prefix_space)
    ]

    # Choose which comparisons to make
    x_column_name: str = "local_estimates_mean"
    y_column_name: str = "loss_mean"

    output_root_dir: pathlib.Path = pathlib.Path(
        embeddings_path_manager.saved_plots_dir_absolute_path,
        "compare_mean_local_estimates_with_mean_losses_for_different_models",
        f"{tokenizer_add_prefix_space=}",
        f"{x_column_name=}_vs_{y_column_name=}",
        main_config.analysis.investigate_distances.get_config_description(),
        main_config.local_estimates.method_description,  # For example: 'twonn'
    )
    output_root_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save descriptive_statistics_df and filtered_descriptive_statistics_df to CSV
    descriptive_statistics_df.to_csv(
        path_or_buf=output_root_dir / "descriptive_statistics_df.csv",
        index=False,
    )
    filtered_descriptive_statistics_df.to_csv(
        path_or_buf=output_root_dir / "filtered_descriptive_statistics_df.csv",
        index=False,
    )

    compare_mean_local_estimates_with_mean_losses_for_different_models(
        descriptive_statistics_df=filtered_descriptive_statistics_df,
        output_root_dir=output_root_dir,
        x_column_name=x_column_name,
        y_column_name=y_column_name,
        verbosity=verbosity,
        logger=logger,
    )

    logger.info(
        msg="Script finished.",
    )


def compare_mean_local_estimates_with_mean_losses_for_different_models(
    descriptive_statistics_df: pd.DataFrame,
    output_root_dir: pathlib.Path,
    x_column_name: str = "local_estimates_mean",
    y_column_name: str = "loss_mean",
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Create plots with compare mean local estimates with mean losses for different models."""
    # # # #
    # Common parameters for all plots
    axes_limits_choices: list[dict] = [
        {
            "x_min": None,
            "x_max": None,
            "y_min": None,
            "y_max": None,
            "output_pdf_width": 2_000,
            "output_pdf_height": 2_000,
        },
        {
            "x_min": 5.5,
            "x_max": 18.5,
            "y_min": 1.0,
            "y_max": 4.5,
            "output_pdf_width": 2_000,
            "output_pdf_height": 2_000,
        },
    ]

    # # # #
    # Log information about the different values in the DataFrame.
    # Notes:
    # - The individual plotting functions iterate over the different values,
    #   they will be computed again there.

    data_full_options: list[str] = descriptive_statistics_df["data_full"].unique().tolist()
    # > Example:
    # > data_subsampling_full = "split=validation_samples=10000_sampling=random_sampling-seed=777"
    data_subsampling_full_options: list[str] = descriptive_statistics_df["data_subsampling_full"].unique().tolist()
    model_partial_name_options: list[str] = descriptive_statistics_df["model_partial_name"].unique().tolist()

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{data_full_options = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"{data_subsampling_full_options = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"{model_partial_name_options = }",  # noqa: G004 - low overhead
        )

    # The identifier of the base model.
    # This value will be used to filter the DataFrame
    # for the correlation analysis and for the model checkpoint analysis.
    base_model_model_partial_name = "model=roberta-base"

    # ========================================================== #
    # Create a common plot for
    # - all datasets
    # - all splits
    # - all models together
    # ========================================================== #

    create_plot_for_all_datasets_all_splits_all_models(
        descriptive_statistics_df=descriptive_statistics_df,
        output_root_dir=output_root_dir,
        x_column_name=x_column_name,
        y_column_name=y_column_name,
        axes_limits_choices=axes_limits_choices,
        verbosity=verbosity,
        logger=logger,
    )

    # ========================================================== #
    # Create separate plots for:
    # - individual splits
    # - all datasets together
    # - all models together
    # ========================================================== #

    create_plots_for_individual_splits_all_datasets_all_models(
        descriptive_statistics_df=descriptive_statistics_df,
        output_root_dir=output_root_dir,
        x_column_name=x_column_name,
        y_column_name=y_column_name,
        axes_limits_choices=axes_limits_choices,
        verbosity=verbosity,
        logger=logger,
    )

    # ========================================================== #
    # Create separate plots for:
    # - individual splits
    # - individual datasets
    # - all models together
    # ========================================================== #

    create_plots_for_individual_splits_individual_datasets_all_models(
        descriptive_statistics_df=descriptive_statistics_df,
        output_root_dir=output_root_dir,
        x_column_name=x_column_name,
        y_column_name=y_column_name,
        axes_limits_choices=axes_limits_choices,
        verbosity=verbosity,
        logger=logger,
    )

    # ========================================================== #
    # Create separate plots for:
    # - individual splits
    # - individual models (one finetuning setup plus base model)
    # - all datasets together
    # ========================================================== #

    create_plots_for_individual_splits_individual_models_all_datasets(
        descriptive_statistics_df=descriptive_statistics_df,
        base_model_model_partial_name=base_model_model_partial_name,
        output_root_dir=output_root_dir,
        x_column_name=x_column_name,
        y_column_name=y_column_name,
        axes_limits_choices=axes_limits_choices,
        verbosity=verbosity,
        logger=logger,
    )


def compute_and_save_correlations_on_filtered_df(
    filtered_df: pd.DataFrame,
    x_column_name: str,
    y_column_name: str,
    output_folder: pathlib.Path,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Compute and save correlations on the filtered DataFrame."""
    filtered_correlations_df: pd.DataFrame = compute_correlations_with_count(
        df=filtered_df,
        cols=[
            x_column_name,
            y_column_name,
        ],
        methods=None,  # 'None' means that default correlation methods are used
        significance_level=0.05,
    )

    if verbosity >= Verbosity.NORMAL:
        log_dataframe_info(
            df=filtered_correlations_df,
            df_name="filtered_correlations_df",
            logger=logger,
        )

    filtered_correlations_df_save_path: pathlib.Path = pathlib.Path(
        output_folder,
        "filtered_correlations_df.csv",
    )
    filtered_correlations_df_save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    filtered_correlations_df.to_csv(
        path_or_buf=filtered_correlations_df_save_path,
        index=False,
    )


@dataclass
class SelectedDataAndComparisonParameters:
    """Data class for selected data and comparison parameters."""

    selected_statistics_df: pd.DataFrame
    x_column_name: str
    y_column_name: str
    output_folder: pathlib.Path

    subtitle_text: str = "placeholder_subtitle_text"


def create_plot_for_all_datasets_all_splits_all_models(
    descriptive_statistics_df: pd.DataFrame,
    output_root_dir: pathlib.Path,
    x_column_name: str,
    y_column_name: str,
    axes_limits_choices: list[dict],
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    descriptive_statistics_df_copy: pd.DataFrame = descriptive_statistics_df.copy()

    output_folder = pathlib.Path(
        output_root_dir,
        "plots_for_all_splits_and_all_datasets_and_all_models",
    )
    subtitle_text: str = "all_splits_and_all_datasets_and_all_models"

    # # # #
    # No filtering in this case
    filtered_df: pd.DataFrame = descriptive_statistics_df_copy
    if verbosity >= Verbosity.NORMAL:
        log_dataframe_info(
            df=filtered_df,
            df_name="filtered_df",
            logger=logger,
        )

    compute_and_save_correlations_on_filtered_df(
        filtered_df=filtered_df,
        x_column_name=x_column_name,
        y_column_name=y_column_name,
        output_folder=output_folder,
        verbosity=verbosity,
        logger=logger,
    )

    # We use the point size to indicate the subsampling.
    # For this to work, we need to create a mapped column that contains the point size.
    size_mapping_dict: dict = {
        "split=train_samples=10000_sampling=take_first": 5,
        "split=validation_samples=10000_sampling=random_sampling-seed=777": 10,
    }
    filtered_df["size_column"] = filtered_df["data_subsampling_full"].map(arg=size_mapping_dict)
    # Fill NaN values with a default value
    filtered_df["size_column"] = filtered_df["size_column"].fillna(
        value=3,
    )

    for axes_limits in axes_limits_choices:
        plot_name: str = (
            f"{x_column_name}_vs_{y_column_name}"
            f"_{axes_limits['x_min']}_{axes_limits['x_max']}"
            f"_{axes_limits['y_min']}_{axes_limits['y_max']}"
        )

        create_scatter_plot(
            df=filtered_df,
            output_folder=output_folder,
            plot_name=plot_name,
            subtitle_text=subtitle_text,
            x_column_name=x_column_name,
            y_column_name=y_column_name,
            color_column_name="data_full",
            symbol_column_name="model_partial_name",
            size_column_name="size_column",
            hover_data=filtered_df.columns.tolist(),
            **axes_limits,
            show_plot=False,
            verbosity=verbosity,
            logger=logger,
        )


def create_plots_for_individual_splits_all_datasets_all_models(
    descriptive_statistics_df: pd.DataFrame,
    output_root_dir: pathlib.Path,
    x_column_name: str,
    y_column_name: str,
    axes_limits_choices: list[dict],
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Create plots for individual splits, all datasets, and all models."""
    data_subsampling_full_options: list[str] = descriptive_statistics_df["data_subsampling_full"].unique().tolist()

    combinations = itertools.product(
        data_subsampling_full_options,
    )

    # Note: If combinations contains only one element, the unpacked value still needs to be put into a tuple.
    for (data_subsampling_full,) in tqdm(
        iterable=combinations,
        desc="Iterating over different plot combinations",
    ):
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"{data_subsampling_full = }",  # noqa: G004 - low overhead
            )

        output_folder = pathlib.Path(
            output_root_dir,
            "plots_for_individual_splits_and_all_datasets_and_all_models",
            f"{data_subsampling_full=}",
        )
        subtitle_text: str = f"{data_subsampling_full=}"

        # # # #
        # Only filter by the subsampling
        descriptive_statistics_df_copy: pd.DataFrame = descriptive_statistics_df.copy()

        filtered_df: pd.DataFrame = descriptive_statistics_df_copy[
            (descriptive_statistics_df_copy["data_subsampling_full"] == data_subsampling_full)
        ]
        if verbosity >= Verbosity.NORMAL:
            log_dataframe_info(
                df=filtered_df,
                df_name="filtered_df",
                logger=logger,
            )

        compute_and_save_correlations_on_filtered_df(
            filtered_df=filtered_df,
            x_column_name=x_column_name,
            y_column_name=y_column_name,
            output_folder=output_folder,
            verbosity=verbosity,
            logger=logger,
        )

        for axes_limits in axes_limits_choices:
            plot_name: str = (
                f"{x_column_name}_vs_{y_column_name}"
                f"_{axes_limits['x_min']}_{axes_limits['x_max']}"
                f"_{axes_limits['y_min']}_{axes_limits['y_max']}"
            )

            create_scatter_plot(
                df=filtered_df,
                output_folder=output_folder,
                plot_name=plot_name,
                subtitle_text=subtitle_text,
                x_column_name=x_column_name,
                y_column_name=y_column_name,
                color_column_name="data_full",
                symbol_column_name="model_partial_name",
                size_column_name=None,
                hover_data=filtered_df.columns.tolist(),
                **axes_limits,
                show_plot=False,
                verbosity=verbosity,
                logger=logger,
            )


def create_plots_for_individual_splits_individual_datasets_all_models(
    descriptive_statistics_df: pd.DataFrame,
    output_root_dir: pathlib.Path,
    x_column_name: str,
    y_column_name: str,
    axes_limits_choices: list[dict],
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Create plots for individual splits, individual datasets, and all models."""
    data_full_options: list[str] = descriptive_statistics_df["data_full"].unique().tolist()
    data_subsampling_full_options: list[str] = descriptive_statistics_df["data_subsampling_full"].unique().tolist()

    combinations = itertools.product(
        data_full_options,
        data_subsampling_full_options,
    )

    # Note: If combinations contains only one element, the unpacked value still needs to be put into a tuple.
    for (
        data_full,
        data_subsampling_full,
    ) in tqdm(
        iterable=combinations,
        desc="Iterating over different plot combinations",
    ):
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"{data_full = }",  # noqa: G004 - low overhead
            )
            logger.info(
                msg=f"{data_subsampling_full = }",  # noqa: G004 - low overhead
            )

        output_folder = pathlib.Path(
            output_root_dir,
            "plots_for_individual_splits_and_individual_datasets_and_all_models",
            f"{data_full=}",
            f"{data_subsampling_full=}",
        )
        subtitle_text: str = f"{data_full=}, {data_subsampling_full=}"

        # Make a copy of the DataFrame so that we do not modify the original DataFrame
        descriptive_statistics_df_copy: pd.DataFrame = descriptive_statistics_df.copy()

        # Filter the DataFrame:
        # - We want to make separate plots for each dataset and split.
        filtered_df: pd.DataFrame = descriptive_statistics_df_copy[
            (descriptive_statistics_df_copy["data_full"] == data_full)
            & (descriptive_statistics_df_copy["data_subsampling_full"] == data_subsampling_full)
        ]
        if verbosity >= Verbosity.NORMAL:
            log_dataframe_info(
                df=filtered_df,
                df_name="filtered_df",
                logger=logger,
            )

        compute_and_save_correlations_on_filtered_df(
            filtered_df=filtered_df,
            x_column_name=x_column_name,
            y_column_name=y_column_name,
            output_folder=output_folder,
            verbosity=verbosity,
            logger=logger,
        )

        # Check that certain columns only contain a unique value
        # This is important for consistency in the plots.
        columns_to_check_for_uniqueness: list[str] = [
            "data_full",
            "data_subsampling_full",
            "embedding_data_handler_full",
            "local_estimates_desc_full",
        ]
        for column_name in columns_to_check_for_uniqueness:
            unique_values: pd.Series = filtered_df[column_name].unique()  # type: ignore - problem with pandas typing
            if len(unique_values) == 0:
                logger.warning(
                    msg=f"Column '{column_name = }' does not contain any values.",  # noqa: G004 - low overhead
                )
            elif len(unique_values) != 1:
                msg: str = f"Column '{column_name = }' does not contain a unique value. Found: {unique_values = }"
                raise ValueError(
                    msg,
                )

        for axes_limits in axes_limits_choices:
            plot_name: str = (
                f"{x_column_name}_vs_{y_column_name}"
                f"_{axes_limits['x_min']}_{axes_limits['x_max']}"
                f"_{axes_limits['y_min']}_{axes_limits['y_max']}"
            )
            # - Use the 'model_checkpoint' column for the color
            # - Use the training data description for the model as the symbol
            create_scatter_plot(
                df=filtered_df,
                output_folder=output_folder,
                plot_name=plot_name,
                subtitle_text=subtitle_text,
                x_column_name=x_column_name,
                y_column_name=y_column_name,
                color_column_name="model_checkpoint",
                symbol_column_name="model_partial_name",
                size_column_name=None,
                hover_data=filtered_df.columns.tolist(),
                **axes_limits,
                show_plot=False,
                verbosity=verbosity,
                logger=logger,
            )


def generate_selected_data_for_individual_splits_individual_models_all_datasets(
    descriptive_statistics_df: pd.DataFrame,
    output_root_dir: pathlib.Path,
    x_column_name: str,
    y_column_name: str,
    base_model_model_partial_name: str,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> Generator[
    SelectedDataAndComparisonParameters,
    None,
    None,
]:
    """Generate selected data for individual splits, individual models, and all datasets."""
    data_subsampling_full_options: list[str] = descriptive_statistics_df["data_subsampling_full"].unique().tolist()
    model_partial_name_options: list[str] = descriptive_statistics_df["model_partial_name"].unique().tolist()

    combinations = itertools.product(
        data_subsampling_full_options,
        model_partial_name_options,
    )

    # Note: If combinations contains only one element, the unpacked value still needs to be put into a tuple.
    for (
        data_subsampling_full,
        model_partial_name,
    ) in tqdm(
        iterable=combinations,
        desc="Iterating over different plot combinations",
    ):
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"{data_subsampling_full = }",  # noqa: G004 - low overhead
            )
            logger.info(
                msg=f"{model_partial_name = }",  # noqa: G004 - low overhead
            )

        output_folder = pathlib.Path(
            output_root_dir,
            "plots_for_individual_splits_and_all_datasets_and_individual_models",
            f"{data_subsampling_full=}",
            f"{model_partial_name=}",
        )
        subtitle_text: str = f"{data_subsampling_full=}; {model_partial_name=}"

        # Make a copy of the DataFrame so that we do not modify the original DataFrame
        descriptive_statistics_df_copy: pd.DataFrame = descriptive_statistics_df.copy()

        # Filter the DataFrame:
        # - We want to make separate plots for each dataset and split.
        filtered_df: pd.DataFrame = descriptive_statistics_df_copy[
            (descriptive_statistics_df_copy["data_subsampling_full"] == data_subsampling_full)
            & (
                (descriptive_statistics_df_copy["model_partial_name"] == model_partial_name)
                | (descriptive_statistics_df_copy["model_partial_name"] == base_model_model_partial_name)
            )
        ]
        if verbosity >= Verbosity.NORMAL:
            log_dataframe_info(
                df=filtered_df,
                df_name="filtered_df",
                logger=logger,
            )

        result = SelectedDataAndComparisonParameters(
            selected_statistics_df=filtered_df,
            x_column_name=x_column_name,
            y_column_name=y_column_name,
            output_folder=output_folder,
            subtitle_text=subtitle_text,
        )

        yield result


def create_plots_for_individual_splits_individual_models_all_datasets(
    descriptive_statistics_df: pd.DataFrame,
    output_root_dir: pathlib.Path,
    x_column_name: str,
    y_column_name: str,
    axes_limits_choices: list[dict],
    base_model_model_partial_name: str = "model=roberta-base",
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Create plots for individual splits, individual models, and all datasets.

    base_model_model_partial_name:
        - The identifier of the base model.
    """
    for selected_data in tqdm(
        iterable=generate_selected_data_for_individual_splits_individual_models_all_datasets(
            descriptive_statistics_df=descriptive_statistics_df,
            output_root_dir=output_root_dir,
            x_column_name=x_column_name,
            y_column_name=y_column_name,
            base_model_model_partial_name=base_model_model_partial_name,
            verbosity=verbosity,
            logger=logger,
        ),
    ):
        compute_and_save_correlations_on_filtered_df(
            filtered_df=selected_data.selected_statistics_df,
            x_column_name=selected_data.x_column_name,
            y_column_name=selected_data.y_column_name,
            output_folder=selected_data.output_folder,
            verbosity=verbosity,
            logger=logger,
        )

        for axes_limits in axes_limits_choices:
            # # # #
            # Call the scatter plot function
            plot_name: str = (
                f"scatterplot"
                f"_{selected_data.x_column_name}_vs_{selected_data.y_column_name}"
                f"_{axes_limits['x_min']}_{axes_limits['x_max']}"
                f"_{axes_limits['y_min']}_{axes_limits['y_max']}"
            )
            # - Use the 'model_checkpoint' column for the color
            # - Use the training data description for the model as the symbol
            create_scatter_plot(
                df=selected_data.selected_statistics_df,
                output_folder=selected_data.output_folder,
                plot_name=plot_name,
                subtitle_text=selected_data.subtitle_text,
                x_column_name=selected_data.x_column_name,
                y_column_name=selected_data.y_column_name,
                color_column_name="model_checkpoint",
                symbol_column_name="data_full",
                size_column_name=None,
                hover_data=selected_data.selected_statistics_df.columns.tolist(),
                **axes_limits,
                show_plot=False,
                verbosity=verbosity,
                logger=logger,
            )

            # # # #
            # Call the line plot function
            line_plot_x_column_name = "model_checkpoint"

            # We want line plots for both x and y columns
            for line_plot_y_column_name in [
                selected_data.x_column_name,
                selected_data.y_column_name,
            ]:
                plot_name: str = (
                    f"lineplot"
                    f"_{line_plot_x_column_name}_vs_{line_plot_y_column_name}"
                    f"_{axes_limits['y_min']}_{axes_limits['y_max']}"
                )

                line_plot_grouped_by_categorical_column(
                    df=selected_data.selected_statistics_df,
                    output_folder=selected_data.output_folder,
                    x_column=line_plot_x_column_name,
                    y_column=line_plot_y_column_name,
                    group_column="data_full",
                    plot_name=plot_name,
                    y_min=axes_limits["y_min"],
                    y_max=axes_limits["y_max"],
                    verbosity=verbosity,
                    logger=logger,
                )


if __name__ == "__main__":
    setup_omega_conf()

    main()

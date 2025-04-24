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

"""Create plots of the local estimates and compare with other task performance measures."""

import logging
import pathlib
from typing import TYPE_CHECKING

import hydra
import omegaconf
import pandas as pd
from tqdm import tqdm

from topollm.analysis.comparisons.compare_mean_estimates_over_different_datasets_and_models.compare_mean_local_estimates_with_mean_losses import (
    generate_selected_data_for_individual_splits_individual_models_all_datasets,
)
from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
from topollm.data_processing.iteration_over_directories.load_json_dicts_from_folder_structure_into_df import (
    load_json_dicts_from_folder_structure_into_df,
)
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.plotting.line_plot_grouped_by_categorical_column import (
    generate_color_mapping,
    line_plot_grouped_by_categorical_column,
)
from topollm.plotting.plot_size_config import PlotColumnsConfig, PlotSizeConfigFlat

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig
    from topollm.path_management.embeddings.protocol import EmbeddingsPathManager
    from topollm.typing.enums import Verbosity

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
    # Load data
    # ================================================== #

    # The following directory contains the precomputed local estimates.
    # Logging of the directory is done in the function which iterates over the directories.
    iteration_root_dir = pathlib.Path(
        embeddings_path_manager.get_local_estimates_root_dir_absolute_path(),
    )

    loaded_df: pd.DataFrame = load_json_dicts_from_folder_structure_into_df(
        iteration_root_dir=iteration_root_dir,
        pattern="**/additional_pointwise_results_statistics.json",
        verbosity=verbosity,
        logger=logger,
    )

    # # #
    # Filter the DataFrame for selected settings
    tokenizer_add_prefix_space = "False"

    filtered_loaded_df: pd.DataFrame = loaded_df.copy()
    filtered_loaded_df = filtered_loaded_df[
        (filtered_loaded_df["tokenizer_add_prefix_space"] == tokenizer_add_prefix_space)
    ]

    # # # #
    # Choose which comparisons to make
    y_column_name: str = "pointwise_results_array_np_np_mean"

    output_root_dir: pathlib.Path = pathlib.Path(
        embeddings_path_manager.saved_plots_dir_absolute_path,
        "task_performance_analysis",
        f"{tokenizer_add_prefix_space=}",
        f"{y_column_name=}",
    )
    output_root_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save the DataFrames to CSV files
    loaded_df.to_csv(
        path_or_buf=output_root_dir / "loaded_df.csv",
        index=False,
    )
    filtered_loaded_df.to_csv(
        path_or_buf=output_root_dir / "filtered_loaded_df.csv",
        index=False,
    )

    # ================================================== #
    # Create plots
    # ================================================== #

    # # # #
    # Common parameters for all plots
    plot_size_configs_list: list[PlotSizeConfigFlat] = [
        PlotSizeConfigFlat(
            x_min=None,
            x_max=None,
            y_min=None,
            y_max=None,
            output_pdf_width=2_000,
            output_pdf_height=1_500,
        ),
        PlotSizeConfigFlat(
            x_min=None,
            x_max=None,
            y_min=1.5,
            y_max=10.5,
            output_pdf_width=2_000,
            output_pdf_height=1_500,
        ),
    ]

    # The identifier of the base model.
    # This value will be used to filter the DataFrame
    # for the correlation analysis and for the model checkpoint analysis.
    base_model_model_partial_name = "model=roberta-base"

    plot_columns_config = PlotColumnsConfig(
        x_column="model_checkpoint",
        y_column=y_column_name,
        group_column="data_full",
        std_column="pointwise_results_array_np_np_std",
    )

    color_mapping: dict = generate_color_mapping(
        df=filtered_loaded_df,
        group_column="data_full",
    )

    for selected_data in tqdm(
        iterable=generate_selected_data_for_individual_splits_individual_models_all_datasets(
            descriptive_statistics_df=filtered_loaded_df,
            output_root_dir=output_root_dir,
            x_column_name=plot_columns_config.x_column,
            y_column_name=plot_columns_config.y_column,
            base_model_model_partial_name=base_model_model_partial_name,
            verbosity=verbosity,
            logger=logger,
        ),
    ):
        for plot_size_config in plot_size_configs_list:
            plot_name: str = (
                f"lineplot"
                f"_{plot_columns_config.x_column}_vs_{plot_columns_config.y_column}"
                f"_{plot_size_config.y_min}_{plot_size_config.y_max}"
            )

            line_plot_grouped_by_categorical_column(
                df=selected_data.selected_statistics_df,
                output_folder=selected_data.output_folder,
                plot_name=plot_name,
                subtitle_text=selected_data.subtitle_text,
                plot_columns_config=plot_columns_config,
                color_mapping=color_mapping,
                plot_size_config=plot_size_config,
                verbosity=verbosity,
                logger=logger,
            )

    logger.info(
        msg="Script finished.",
    )


if __name__ == "__main__":
    main()

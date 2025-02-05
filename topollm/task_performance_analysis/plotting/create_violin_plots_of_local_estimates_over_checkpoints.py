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

import itertools
import logging
import pathlib
from typing import TYPE_CHECKING

import hydra
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import pandas as pd
from tqdm import tqdm

from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
from topollm.config_classes.setup_OmegaConf import setup_omega_conf
from topollm.data_processing.dictionary_handling import (
    dictionary_to_partial_path,
    filter_list_of_dictionaries_by_key_value_pairs,
)
from topollm.data_processing.iteration_over_directories.load_np_arrays_from_folder_structure_into_list_of_dicts import (
    load_np_arrays_from_folder_structure_into_list_of_dicts,
)
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.log_array_info import log_array_info
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.plotting.line_plot_grouped_by_categorical_column import (
    PlotColumnsConfig,
    PlotSizeConfig,
    generate_color_mapping,
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
    # Load data
    # ================================================== #

    # The following directory contains the precomputed local estimates.
    # Logging of the directory is done in the function which iterates over the directories.
    # TODO: We write this function first to iterate only over one of the subfolders.
    iteration_root_dir = pathlib.Path(
        embeddings_path_manager.get_local_estimates_root_dir_absolute_path(),
        "data=sgd_rm-empty=True_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags",
        "split=validation_samples=10000_sampling=random_sampling-seed=778",
        "edh-mode=regular_lvl=token/add-prefix-space=False_max-len=512",
    )

    loaded_data: list[dict] = load_np_arrays_from_folder_structure_into_list_of_dicts(
        iteration_root_dir=iteration_root_dir,
        pattern="**/local_estimates_pointwise_array.npy",
        verbosity=verbosity,
        logger=logger,
    )

    # The identifier of the base model.
    # This value will be used to filter the DataFrame
    # for the correlation analysis and for the model checkpoint analysis.
    base_model_model_partial_name = "model=roberta-base"

    # TODO: Add the base model to the plots

    # # # #
    # Choose which comparisons to make
    array_key_name: str = "file_data"

    output_root_dir: pathlib.Path = pathlib.Path(
        embeddings_path_manager.saved_plots_dir_absolute_path,
        "task_performance_analysis",
        "distribution_of_local_estimates",
    )
    output_root_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ================================================== #
    # Create plots
    # ================================================== #

    # # # #
    # Common parameters for all plots
    plot_size_configs_list: list[PlotSizeConfig] = [
        PlotSizeConfig(
            x_min=None,
            x_max=None,
            y_min=None,
            y_max=None,
            output_pdf_width=2_000,
            output_pdf_height=1_500,
        ),
        PlotSizeConfig(
            x_min=None,
            x_max=None,
            y_min=0.0,
            y_max=10.5,
            output_pdf_width=2_000,
            output_pdf_height=1_500,
        ),
    ]

    data_full_options: set[str] = {single_dict["data_full"] for single_dict in loaded_data}
    data_subsampling_full_options: set[str] = {single_dict["data_subsampling_full"] for single_dict in loaded_data}
    model_partial_name_options: set[str] = {single_dict["model_partial_name"] for single_dict in loaded_data}

    # # # #
    # Create plots which show the distribution of the local estimates over the checkpoints

    for (
        data_full,
        data_subsampling_full,
        model_partial_name,
    ) in tqdm(
        iterable=itertools.product(
            data_full_options,
            data_subsampling_full_options,
            model_partial_name_options,
        ),
        desc="Plotting different choices",
    ):
        filter_key_value_pairs: dict = {
            "tokenizer_add_prefix_space": "False",  # This needs to be a string
            "data_full": data_full,
            "data_subsampling_full": data_subsampling_full,
            "model_partial_name": model_partial_name,
        }

        filtered_data: list[dict] = filter_list_of_dictionaries_by_key_value_pairs(
            list_of_dicts=loaded_data,
            key_value_pairs=filter_key_value_pairs,
        )

        if len(filtered_data) == 0:
            logger.warning(
                msg=f"No data found for {filter_key_value_pairs = }.",  # noqa: G004 - low overhead
            )
            logger.warning(
                msg="Skipping this combination of parameters.",
            )
            continue

        extracted_arrays: list[np.ndarray] = [single_dict[array_key_name] for single_dict in filtered_data]
        model_checkpoint_str_list: list[str] = [
            str(object=single_dict["model_checkpoint"]) for single_dict in filtered_data
        ]

        plots_output_dir: pathlib.Path = pathlib.Path(
            output_root_dir,
            "plots_over_checkpoints",
            dictionary_to_partial_path(
                dictionary=filter_key_value_pairs,
            ),
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"{plots_output_dir = }",  # noqa: G004 - low overhead
            )

        # TODO: Sort the arrays by increasing model checkpoint
        pass  # This is here for setting a breakpoint

        for plot_size_config in plot_size_configs_list:
            # # # #
            # Violin plots
            (
                fig,
                ax,
            ) = plt.subplots(
                figsize=(
                    plot_size_config.output_pdf_width / 100,
                    plot_size_config.output_pdf_height / 100,
                ),
            )

            # plot violin plot

            ax.violinplot(
                dataset=extracted_arrays,
                showmeans=True,
                showmedians=True,
            )
            ax.set_title(
                label="Violin plot",
            )

            # adding horizontal grid lines
            ax.yaxis.grid(
                visible=True,
            )

            # Use the model checkpoints to set the xticks
            ax.set_xticks(
                ticks=[y + 1 for y in range(len(extracted_arrays))],
                labels=model_checkpoint_str_list,
            )

            ax.set_xlabel(
                xlabel="Checkpoints",
            )
            ax.set_ylabel(
                ylabel="Observed values",
            )

            # Set the y-axis limits
            if plot_size_config.y_min is not None:
                ax.set_ylim(
                    bottom=plot_size_config.y_min,
                )
            if plot_size_config.y_max is not None:
                ax.set_ylim(
                    top=plot_size_config.y_max,
                )

            plot_name: str = f"violinplot" f"_{plot_size_config.y_min}_{plot_size_config.y_max}"
            plot_output_path: pathlib.Path = pathlib.Path(
                plots_output_dir,
                f"{plot_name}.pdf",
            )
            plot_output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"Saving plot to {plot_output_path = } ...",  # noqa: G004 - low overhead
                )
            fig.savefig(
                fname=plot_output_path,
                bbox_inches="tight",
            )
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"Saving plot to {plot_output_path = } DONE",  # noqa: G004 - low overhead
                )

            # # # #
            # Boxplots
            fig, ax = plt.subplots(
                figsize=(
                    plot_size_config.output_pdf_width / 100,
                    plot_size_config.output_pdf_height / 100,
                ),
            )

            # plot box plot
            ax.boxplot(
                x=extracted_arrays,
            )
            ax.set_title(
                label="Box plot",
            )

            # adding horizontal grid lines
            ax.yaxis.grid(
                visible=True,
            )

            # Use the model checkpoints to set the xticks
            ax.set_xticks(
                ticks=[y + 1 for y in range(len(extracted_arrays))],
                labels=model_checkpoint_str_list,
            )

            ax.set_xlabel(
                xlabel="Checkpoints",
            )
            ax.set_ylabel(
                ylabel="Observed values",
            )

            # Set the y-axis limits
            if plot_size_config.y_min is not None:
                ax.set_ylim(
                    bottom=plot_size_config.y_min,
                )
            if plot_size_config.y_max is not None:
                ax.set_ylim(
                    top=plot_size_config.y_max,
                )

            plot_name: str = f"boxplot" f"_{plot_size_config.y_min}_{plot_size_config.y_max}"
            plot_output_path: pathlib.Path = pathlib.Path(
                plots_output_dir,
                f"{plot_name}.pdf",
            )
            plot_output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"Saving plot to {plot_output_path = } ...",  # noqa: G004 - low overhead
                )
            fig.savefig(
                fname=plot_output_path,
                bbox_inches="tight",
            )
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"Saving plot to {plot_output_path = } DONE",  # noqa: G004 - low overhead
                )

    # # # #
    # Create plots which show the distribution of the local estimates over different layers of the model

    # TODO: Implement the creation of the plots over the layers

    logger.info(
        msg="Script finished.",
    )


if __name__ == "__main__":
    setup_omega_conf()

    main()

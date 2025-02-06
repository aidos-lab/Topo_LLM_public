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
from dataclasses import dataclass
from typing import TYPE_CHECKING

import hydra
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
from tqdm import tqdm

from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
from topollm.config_classes.setup_OmegaConf import setup_omega_conf
from topollm.data_processing.dictionary_handling import (
    dictionary_to_partial_path,
    filter_list_of_dictionaries_by_key_value_pairs,
    generate_fixed_parameters_text_from_dict,
)
from topollm.data_processing.iteration_over_directories.load_np_arrays_from_folder_structure_into_list_of_dicts import (
    load_np_arrays_from_folder_structure_into_list_of_dicts,
)
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.plotting.line_plot_grouped_by_categorical_column import (
    PlotColumnsConfig,
    PlotSizeConfig,
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
    iteration_root_dir = pathlib.Path(
        embeddings_path_manager.get_local_estimates_root_dir_absolute_path(),
    )

    pattern = (
        "**/"
        "split=validation_samples=10000_sampling=random_sampling-seed=778/"
        "edh-mode=regular_lvl=token/"
        "add-prefix-space=False_max-len=512/"
        "**/"
        "local_estimates_pointwise_array.npy"
    )

    loaded_data: list[dict] = load_np_arrays_from_folder_structure_into_list_of_dicts(
        iteration_root_dir=iteration_root_dir,
        pattern=pattern,
        verbosity=verbosity,
        logger=logger,
    )

    # The identifier of the base model.
    # This value will be used to filter the DataFrame
    # for the correlation analysis and for the model checkpoint analysis.
    base_model_model_partial_name = "model=roberta-base"

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

    # # # #
    # Create plots which show the distribution of the local estimates over the checkpoints
    if main_config.feature_flags.task_performance_analysis.plotting_create_distribution_plots_over_model_checkpoints:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Creating the distribution plots over the model checkpoints ...",
            )
        create_distribution_plots_over_model_checkpoints(
            loaded_data=loaded_data,
            base_model_model_partial_name=base_model_model_partial_name,
            array_key_name=array_key_name,
            output_root_dir=output_root_dir,
            plot_size_configs_list=plot_size_configs_list,
            verbosity=verbosity,
            logger=logger,
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Creating the distribution plots over the model checkpoints DONE",
            )
    else:
        logger.info(
            msg="Skipping the creation of the distribution plots over the model checkpoints.",
        )

    # # # #
    # Create plots which show the distribution of the local estimates over different layers of the model
    if main_config.feature_flags.task_performance_analysis.plotting_create_distribution_plots_over_model_layers:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Creating the distribution plots over the model layers ...",
            )

        # TODO: Make the layer-wise plots into a function.

        data_full_options: set[str] = {single_dict["data_full"] for single_dict in loaded_data}
        data_subsampling_full_options: set[str] = {single_dict["data_subsampling_full"] for single_dict in loaded_data}
        model_partial_name_options: set[str] = {single_dict["model_partial_name"] for single_dict in loaded_data}
        model_checkpoint_options: set = {single_dict["model_checkpoint"] for single_dict in loaded_data}

        for (
            data_full,
            data_subsampling_full,
            model_partial_name,
            model_checkpoint,
        ) in tqdm(
            iterable=itertools.product(
                data_full_options,
                data_subsampling_full_options,
                model_partial_name_options,
                model_checkpoint_options,
            ),
            desc="Plotting different choices",
        ):
            filter_key_value_pairs: dict = {
                "tokenizer_add_prefix_space": "False",  # This needs to be a string
                "data_full": data_full,
                "data_subsampling_full": data_subsampling_full,
                "model_partial_name": model_partial_name,
                "model_checkpoint": model_checkpoint,
                # We want all checkpoints for the given model checkpoint.
            }
            fixed_params_text: str = generate_fixed_parameters_text_from_dict(
                filters_dict=filter_key_value_pairs,
            )

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

            # Sort the arrays by increasing model layer.
            sorted_data: list[dict] = sorted(
                filtered_data,
                key=lambda single_dict: int(single_dict["model_layer"]),
            )

            extracted_arrays: list[np.ndarray] = [single_dict[array_key_name] for single_dict in sorted_data]
            model_layer_str_list: list[str] = [str(object=single_dict["model_layer"]) for single_dict in sorted_data]

            plots_output_dir: pathlib.Path = pathlib.Path(
                output_root_dir,
                "plots_over_layers",
                dictionary_to_partial_path(
                    dictionary=filter_key_value_pairs,
                ),
            )
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"{plots_output_dir = }",  # noqa: G004 - low overhead
                )

            ticks_and_labels: TicksAndLabels = TicksAndLabels(
                xlabel="layers",
                ylabel=array_key_name,
                xticks_labels=model_layer_str_list,
            )

            for plot_size_config in plot_size_configs_list:
                # # # #
                # Violin plots
                make_distribution_violinplots_from_extracted_arrays(
                    extracted_arrays=extracted_arrays,
                    ticks_and_labels=ticks_and_labels,
                    fixed_params_text=fixed_params_text,
                    plots_output_dir=plots_output_dir,
                    plot_size_config=plot_size_config,
                    verbosity=verbosity,
                    logger=logger,
                )

                # # # #
                # Boxplots
                make_distribution_boxplots_from_extracted_arrays(
                    extracted_arrays=extracted_arrays,
                    ticks_and_labels=ticks_and_labels,
                    fixed_params_text=fixed_params_text,
                    plots_output_dir=plots_output_dir,
                    plot_size_config=plot_size_config,
                    verbosity=verbosity,
                    logger=logger,
                )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Creating the distribution plots over the model layers DONE",
            )
    else:
        logger.info(
            msg="Skipping the creation of the distribution plots over the model layers.",
        )

    logger.info(
        msg="Script finished.",
    )


def create_distribution_plots_over_model_checkpoints(
    loaded_data: list[dict],
    base_model_model_partial_name: str,
    array_key_name: str,
    output_root_dir: pathlib.Path,
    plot_size_configs_list: list[PlotSizeConfig],
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Create plots which show the distribution of the local estimates over the checkpoints."""
    data_full_options: set[str] = {single_dict["data_full"] for single_dict in loaded_data}
    data_subsampling_full_options: set[str] = {single_dict["data_subsampling_full"] for single_dict in loaded_data}
    model_partial_name_options: set[str] = {single_dict["model_partial_name"] for single_dict in loaded_data}

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
            "model_layer": -1,  # This needs to be an integer
            "model_partial_name": model_partial_name,
        }
        fixed_params_text: str = generate_fixed_parameters_text_from_dict(
            filters_dict=filter_key_value_pairs,
        )

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

        # Extract the dictionary which matches the same other parameters but the base model.
        filter_key_value_pairs_base_model: dict = {
            **filter_key_value_pairs,
            "model_partial_name": base_model_model_partial_name,
        }
        filtered_data_base_model: list[dict] = filter_list_of_dictionaries_by_key_value_pairs(
            list_of_dicts=loaded_data,
            key_value_pairs=filter_key_value_pairs_base_model,
        )
        if len(filtered_data_base_model) == 0:
            logger.warning(
                msg=f"No base model data found for {filter_key_value_pairs = }.",  # noqa: G004 - low overhead
            )
            logger.warning(
                msg="Proceeding without adding base model data.",
            )
        elif len(filtered_data_base_model) == 1:
            logger.info(
                msg=f"Unique base model data found for {filter_key_value_pairs = }.",  # noqa: G004 - low overhead
            )
            filtered_data_base_model_dict: dict = filtered_data_base_model[0]
            # Add the base model data to the list of data to plot.
            filtered_data.append(
                filtered_data_base_model_dict,
            )
        elif len(filtered_data_base_model) > 1:
            logger.warning(
                f"Ambiguous base model data ({len(filtered_data_base_model)} entries) "  # noqa: G004 - low overhead
                f"found for {filter_key_value_pairs = }.",
            )
            logger.warning(
                msg="Will use the first entry.",
            )
            filtered_data_base_model_dict: dict = filtered_data_base_model[0]
            # Add the base model data to the list of data to plot.
            filtered_data.append(
                filtered_data_base_model_dict,
            )

        # Sort the arrays by increasing model checkpoint.
        # Then from this point, the list of arrays and list of extracted checkpoints will be in the correct order.
        # 1. Step: Replace None model checkpoints with -1.
        for single_dict in filtered_data:
            if single_dict["model_checkpoint"] is None:
                single_dict["model_checkpoint"] = -1
        # 2. Step: Call sorting function.
        sorted_data: list[dict] = sorted(
            filtered_data,
            key=lambda single_dict: int(single_dict["model_checkpoint"]),
        )

        extracted_arrays: list[np.ndarray] = [single_dict[array_key_name] for single_dict in sorted_data]
        model_checkpoint_str_list: list[str] = [
            str(object=single_dict["model_checkpoint"]) for single_dict in sorted_data
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

        ticks_and_labels: TicksAndLabels = TicksAndLabels(
            xlabel="checkpoints",
            ylabel=array_key_name,
            xticks_labels=model_checkpoint_str_list,
        )

        for plot_size_config in plot_size_configs_list:
            # # # #
            # Violin plots
            make_distribution_violinplots_from_extracted_arrays(
                extracted_arrays=extracted_arrays,
                ticks_and_labels=ticks_and_labels,
                fixed_params_text=fixed_params_text,
                plots_output_dir=plots_output_dir,
                plot_size_config=plot_size_config,
                verbosity=verbosity,
                logger=logger,
            )

            # # # #
            # Boxplots
            make_distribution_boxplots_from_extracted_arrays(
                extracted_arrays=extracted_arrays,
                ticks_and_labels=ticks_and_labels,
                fixed_params_text=fixed_params_text,
                plots_output_dir=plots_output_dir,
                plot_size_config=plot_size_config,
                verbosity=verbosity,
                logger=logger,
            )


@dataclass
class TicksAndLabels:
    """Container for ticks and labels."""

    xlabel: str
    ylabel: str
    xticks_labels: list[str]


def make_distribution_violinplots_from_extracted_arrays(
    extracted_arrays: list[np.ndarray],
    ticks_and_labels: TicksAndLabels,
    fixed_params_text: str,
    plots_output_dir: pathlib.Path,
    plot_size_config: PlotSizeConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    (
        fig,
        ax,
    ) = plt.subplots(
        figsize=(
            plot_size_config.output_pdf_width / 100,
            plot_size_config.output_pdf_height / 100,
        ),
    )

    # Plot violin plot
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
        labels=ticks_and_labels.xticks_labels,
    )

    ax.set_xlabel(
        xlabel=ticks_and_labels.xlabel,
    )
    ax.set_ylabel(
        ylabel=ticks_and_labels.ylabel,
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

    if fixed_params_text is not None:
        ax.text(
            x=1.05,
            y=0.25,
            s=f"Fixed Parameters:\n{fixed_params_text}",
            transform=plt.gca().transAxes,
            fontsize=6,
            verticalalignment="top",
            bbox={
                "boxstyle": "round",
                "facecolor": "wheat",
                "alpha": 0.3,
            },
        )

        # Saving the plot
    plot_name: str = f"violinplot_{plot_size_config.y_min}_{plot_size_config.y_max}"
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


def make_distribution_boxplots_from_extracted_arrays(
    extracted_arrays: list[np.ndarray],
    ticks_and_labels: TicksAndLabels,
    fixed_params_text: str,
    plots_output_dir: pathlib.Path,
    plot_size_config: PlotSizeConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Create a boxplot."""
    (
        fig,
        ax,
    ) = plt.subplots(
        figsize=(
            plot_size_config.output_pdf_width / 100,
            plot_size_config.output_pdf_height / 100,
        ),
    )

    # Plot box plot
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
        labels=ticks_and_labels.xticks_labels,
    )

    ax.set_xlabel(
        xlabel=ticks_and_labels.xlabel,
    )
    ax.set_ylabel(
        ylabel=ticks_and_labels.ylabel,
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

    if fixed_params_text is not None:
        ax.text(
            x=1.05,
            y=0.25,
            s=f"Fixed Parameters:\n{fixed_params_text}",
            transform=plt.gca().transAxes,
            fontsize=6,
            verticalalignment="top",
            bbox={
                "boxstyle": "round",
                "facecolor": "wheat",
                "alpha": 0.3,
            },
        )

    plot_name: str = f"boxplot_{plot_size_config.y_min}_{plot_size_config.y_max}"
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


if __name__ == "__main__":
    setup_omega_conf()

    main()

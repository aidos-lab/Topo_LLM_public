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
import pickle
from typing import TYPE_CHECKING

import hydra
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
from topollm.path_management.embeddings.protocol import EmbeddingsPathManager
from topollm.plotting.line_plot_grouped_by_categorical_column import (
    PlotSizeConfig,
)
from topollm.task_performance_analysis.plotting.distribution_violinplots_and_distribution_boxplots import (
    TicksAndLabels,
    make_distribution_boxplots_from_extracted_arrays,
    make_distribution_violinplots_from_extracted_arrays,
)
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig

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

    patterns_to_iterate_over: list[str] = [
        # # > Splits for the SetSUMBT saved dataloaders
        # (
        #     "**/"
        #     "split=train_samples=10000_sampling=random_sampling-seed=778/"
        #     "edh-mode=regular_lvl=token/"
        #     "add-prefix-space=False_max-len=512/"
        #     "**/"
        #     "local_estimates_pointwise_array.npy"
        # ),
        (
            "**/"
            "split=dev_samples=10000_sampling=random_sampling-seed=778/"
            "edh-mode=regular_lvl=token/"
            "add-prefix-space=False_max-len=512/"
            "**/"
            "local_estimates_pointwise_array.npy"
        ),
        # (
        #     "**/"
        #     "split=test_samples=10000_sampling=random_sampling-seed=778/"
        #     "edh-mode=regular_lvl=token/"
        #     "add-prefix-space=False_max-len=512/"
        #     "**/"
        #     "local_estimates_pointwise_array.npy"
        # ),
        # # > Splits for the Huggingface datasets
        # (
        #     "**/"
        #     "split=validation_samples=10000_sampling=random_sampling-seed=778/"
        #     "edh-mode=regular_lvl=token/"
        #     "add-prefix-space=False_max-len=512/"
        #     "**/"
        #     "local_estimates_pointwise_array.npy"
        # ),
        # # > Other selected datasets and splits
        # (
        #     "**/"
        #     "data=sgd_rm-empty=True_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags/"
        #     "split=validation_samples=10000_sampling=random_sampling-seed=777/"
        #     "edh-mode=masked_token_lvl=token/"
        #     "add-prefix-space=False_max-len=512/"
        #     "**/"
        #     "local_estimates_pointwise_array.npy"
        # ),
    ]

    for pattern in tqdm(
        iterable=patterns_to_iterate_over,
        desc="Iterate over patterns",
    ):
        create_plots_for_given_pattern(
            iteration_root_dir=iteration_root_dir,
            pattern=pattern,
            embeddings_path_manager=embeddings_path_manager,
            do_create_distribution_plots_over_model_checkpoints=main_config.feature_flags.task_performance_analysis.plotting_create_distribution_plots_over_model_checkpoints,
            do_create_distribution_plots_over_model_layers=main_config.feature_flags.task_performance_analysis.plotting_create_distribution_plots_over_model_layers,
            verbosity=verbosity,
            logger=logger,
        )


def create_plots_for_given_pattern(
    iteration_root_dir: pathlib.Path,
    pattern: str,
    embeddings_path_manager: EmbeddingsPathManager,
    *,
    do_create_distribution_plots_over_model_checkpoints: bool = True,
    do_create_distribution_plots_over_model_layers: bool = True,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Create plots for a given pattern."""
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

    output_root_dir: pathlib.Path = (
        embeddings_path_manager.get_saved_plots_distribution_of_local_estimates_dir_absolute_path()
    )
    output_root_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{output_root_dir = }",  # noqa: G004 - low overhead
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
    if do_create_distribution_plots_over_model_checkpoints:
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
    if do_create_distribution_plots_over_model_layers:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Creating the distribution plots over the model layers ...",
            )
        create_distribution_plots_over_model_layers(
            loaded_data=loaded_data,
            array_key_name=array_key_name,
            output_root_dir=output_root_dir,
            plot_size_configs_list=plot_size_configs_list,
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


def create_distribution_plots_over_model_layers(
    loaded_data: list[dict],
    array_key_name: str,
    output_root_dir: pathlib.Path,
    plot_size_configs_list: list[PlotSizeConfig],
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Create plots which show the distribution of the local estimates over the layers."""
    data_full_options: set[str] = {single_dict["data_full"] for single_dict in loaded_data}
    data_subsampling_full_options: set[str] = {single_dict["data_subsampling_full"] for single_dict in loaded_data}
    model_checkpoint_options: set = {single_dict["model_checkpoint"] for single_dict in loaded_data}
    model_partial_name_options: set[str] = {single_dict["model_partial_name"] for single_dict in loaded_data}
    model_seed_options: set = {single_dict["model_seed"] for single_dict in loaded_data}

    for (
        data_full,
        data_subsampling_full,
        model_checkpoint,
        model_partial_name,
        model_seed,
    ) in tqdm(
        iterable=itertools.product(
            data_full_options,
            data_subsampling_full_options,
            model_checkpoint_options,
            model_partial_name_options,
            model_seed_options,
        ),
        desc="Plotting different choices",
    ):
        filter_key_value_pairs: dict = {
            "tokenizer_add_prefix_space": "False",  # This needs to be a string
            "data_full": data_full,
            "data_subsampling_full": data_subsampling_full,
            "model_checkpoint": model_checkpoint,
            # We want all layers for the given model checkpoint and thus, no filtering for model_layer in this case.
            "model_partial_name": model_partial_name,
            "model_seed": model_seed,
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


def create_distribution_plots_over_model_checkpoints(
    loaded_data: list[dict],
    base_model_model_partial_name: str,
    array_key_name: str,
    output_root_dir: pathlib.Path,
    plot_size_configs_list: list[PlotSizeConfig],
    *,
    save_sorted_data_list_of_dicts_with_arrays: bool = True,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Create plots which show the distribution of the local estimates over the checkpoints."""
    data_full_options: set[str] = {single_dict["data_full"] for single_dict in loaded_data}
    data_subsampling_full_options: set[str] = {single_dict["data_subsampling_full"] for single_dict in loaded_data}
    model_layer_options: set = {single_dict["model_layer"] for single_dict in loaded_data}
    model_partial_name_options: set[str] = {single_dict["model_partial_name"] for single_dict in loaded_data}
    model_seed_options: set = {single_dict["model_seed"] for single_dict in loaded_data}

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{data_full_options = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"{data_subsampling_full_options = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"{model_layer_options = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"{model_partial_name_options = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"{model_seed_options = }",  # noqa: G004 - low overhead
        )

    iterable = itertools.product(
        data_full_options,
        data_subsampling_full_options,
        model_layer_options,
        model_partial_name_options,
        model_seed_options,
    )

    total_combinations: int = (
        len(data_full_options)
        * len(data_subsampling_full_options)
        * len(model_layer_options)
        * len(model_partial_name_options)
        * len(model_seed_options)
    )
    for (
        data_full,
        data_subsampling_full,
        model_layer,
        model_partial_name,
        model_seed,
    ) in tqdm(
        iterable=iterable,
        total=total_combinations,
        desc="Plotting different choices",
    ):
        filter_key_value_pairs: dict[
            str,
            str | int,
        ] = {
            "tokenizer_add_prefix_space": "False",  # tokenizer_add_prefix_space needs to be a string
            "data_full": data_full,
            "data_subsampling_full": data_subsampling_full,
            "model_layer": model_layer,  # model_layer needs to be an integer
            "model_partial_name": model_partial_name,
            "model_seed": model_seed,
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

        filtered_data = add_base_model_data(
            loaded_data=loaded_data,
            base_model_model_partial_name=base_model_model_partial_name,
            filter_key_value_pairs=filter_key_value_pairs,
            filtered_data=filtered_data,
            logger=logger,
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

        plots_output_dir: pathlib.Path = construct_plots_over_checkpoints_output_dir_from_filter_key_value_pairs(
            output_root_dir=output_root_dir,
            filter_key_value_pairs=filter_key_value_pairs,
            verbosity=verbosity,
            logger=logger,
        )

        if save_sorted_data_list_of_dicts_with_arrays:
            # Save the sorted data list of dicts with the arrays to a pickle file.
            data_to_save: list[dict] = sorted_data

            sorted_data_output_file_path: pathlib.Path = pathlib.Path(
                plots_output_dir,
                "sorted_data_list_of_dicts_with_arrays.pkl",
            )

            with sorted_data_output_file_path.open(
                mode="wb",
            ) as file:
                pickle.dump(
                    obj=data_to_save,
                    file=file,
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


def construct_plots_over_checkpoints_output_dir_from_filter_key_value_pairs(
    output_root_dir: pathlib.Path,
    filter_key_value_pairs: dict[
        str,
        str | int,
    ],
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> pathlib.Path:
    """Construct the plots output directory from the filter key-value pairs."""
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

    return plots_output_dir


def add_base_model_data(
    loaded_data: list[dict],
    base_model_model_partial_name: str,
    filter_key_value_pairs: dict[
        str,
        str | int,
    ],
    filtered_data: list[dict],
    logger: logging.Logger = default_logger,
) -> list[dict]:
    """Add the base model data to the list of data to plot."""
    # Build the dictionary which matches the same parameters as in the filter
    # but has the model_partial_name from the base model.
    filter_key_value_pairs_base_model: dict = {
        **filter_key_value_pairs,
        "model_partial_name": base_model_model_partial_name,
        "model_seed": None,  # The base model always has model_seed=None.
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

    return filtered_data


if __name__ == "__main__":
    setup_omega_conf()

    main()

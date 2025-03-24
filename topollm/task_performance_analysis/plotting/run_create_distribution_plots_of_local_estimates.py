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
from tqdm import tqdm

from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
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
from topollm.task_performance_analysis.plotting.create_distribution_plots_over_model_checkpoints import (
    create_distribution_plots_over_model_checkpoints,
)
from topollm.task_performance_analysis.plotting.create_distribution_plots_over_model_layers import (
    create_distribution_plots_over_model_layers,
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
        # # # > Splits for the SetSUMBT saved dataloaders
        # (
        #     "**/"
        #     "split=train_samples=10000_sampling=random_sampling-seed=778/"
        #     "edh-mode=regular_lvl=token/"
        #     "add-prefix-space=False_max-len=512/"
        #     "**/"
        #     "local_estimates_pointwise_array.npy"
        # ),
        # (
        #     "**/"
        #     "split=dev_samples=10000_sampling=random_sampling-seed=778/"
        #     "edh-mode=regular_lvl=token/"
        #     "add-prefix-space=False_max-len=512/"
        #     "**/"
        #     "local_estimates_pointwise_array.npy"
        # ),
        # (
        #     "**/"
        #     "split=test_samples=10000_sampling=random_sampling-seed=778/"
        #     "edh-mode=regular_lvl=token/"
        #     "add-prefix-space=False_max-len=512/"
        #     "**/"
        #     "local_estimates_pointwise_array.npy"
        # ),
        # # # > Splits for the Trippy saved dataloaders
        # (
        #     "**/"
        #     "data=trippy_dataloaders_processed_multiwoz21_rm-empty=True_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags/"
        #     "split=*_samples=10000_sampling=random_sampling-seed=778/"
        #     "edh-mode=regular_lvl=token/"
        #     "add-prefix-space=False_max-len=512/"
        #     "**/"
        #     "local_estimates_pointwise_array.npy"
        # ),
        # # > Splits for the Trippy-R saved dataloaders
        (
            "**/"
            "data=trippy_r_dataloaders_processed_multiwoz21_rm-empty=True_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags/"
            "split=*_samples=10000_sampling=random_sampling-seed=778/"
            "edh-mode=regular_lvl=token/"
            "add-prefix-space=False_max-len=512/"
            "**/"
            "local_estimates_pointwise_array.npy"
        ),
        # # # > Splits for saved ERC model dataloaders
        # (
        #     "**/"
        #     "data=ertod_emowoz_*/"
        #     "split=*_samples=10000_sampling=random_sampling-seed=778/"
        #     "edh-mode=regular_lvl=token/"
        #     "add-prefix-space=False_max-len=512/"
        #     "**/"
        #     "local_estimates_pointwise_array.npy"
        # ),
        # # # > Splits for the Huggingface datasets
        # (
        #     "**/"
        #     "split=validation_samples=10000_sampling=random_sampling-seed=778/"
        #     "edh-mode=regular_lvl=token/"
        #     "add-prefix-space=False_max-len=512/"
        #     "**/"
        #     "local_estimates_pointwise_array.npy"
        # ),
        # # # > Other selected datasets and splits
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


if __name__ == "__main__":
    main()

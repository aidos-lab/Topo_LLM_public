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
from topollm.config_classes.setup_OmegaConf import setup_omega_conf
from topollm.data_processing.iteration_over_directories.load_json_dicts_from_folder_structure_into_df import (
    load_json_dicts_from_folder_structure_into_df,
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

    # # #
    # Filter the DataFrame for selected settings
    tokenizer_add_prefix_space = "False"

    # TODO: Implement filtering of the loaded data

    # # # #
    # Choose which comparisons to make
    array_key_name: str = "file_data"

    output_root_dir: pathlib.Path = pathlib.Path(
        embeddings_path_manager.saved_plots_dir_absolute_path,
        "task_performance_analysis",
        f"{tokenizer_add_prefix_space=}",
    )
    output_root_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ================================================== #
    # Create plots
    # ================================================== #

    # TODO: Continue editing the script from here

    example_array = loaded_data[0][array_key_name]

    if verbosity >= Verbosity.NORMAL:
        log_array_info(
            array_=example_array,
            array_name=example_array,
            logger=logger,
        )

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

    logger.info(
        msg="Script finished.",
    )


if __name__ == "__main__":
    setup_omega_conf()

    main()

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

"""Run script to compare computed Hausdorff distances with the local estimates."""

import logging
import pathlib
import pprint
from collections.abc import Generator
from typing import TYPE_CHECKING

import hydra
import hydra.core.hydra_config
import joblib
import omegaconf
import pandas as pd
from tqdm import tqdm

from topollm.analysis.compare_sampling_methods.analysis_modes.checkpoint_analysis_modes import (
    CheckpointAnalysisModes,
)
from topollm.analysis.compare_sampling_methods.analysis_modes.noise_analysis_modes import (
    NoiseAnalysisCombination,
    NoiseAnalysisModes,
)
from topollm.analysis.compare_sampling_methods.checkpoint_analysis.model_checkpoint_analysis import (
    run_checkpoint_analysis_over_different_data_and_models,
)
from topollm.analysis.compare_sampling_methods.checkpoint_analysis.model_loss_extractor import ModelLossExtractor
from topollm.analysis.compare_sampling_methods.extract_results_from_directory_structure import (
    run_search_on_single_base_directory_and_process_and_save,
)
from topollm.analysis.compare_sampling_methods.filter_dataframe_based_on_filters_dict import (
    filter_dataframe_based_on_filters_dict,
)
from topollm.analysis.compare_sampling_methods.load_and_concatenate_saved_dataframes import (
    load_and_concatenate_saved_dataframes,
)
from topollm.analysis.compare_sampling_methods.make_plots import (
    PlotProperties,
    generate_fixed_params_text,
    scatterplot_individual_seed_combinations_and_combined,
)
from topollm.analysis.compare_sampling_methods.organize_results_directory_structure import (
    build_results_directory_structure,
)
from topollm.analysis.compare_sampling_methods.sensitivity_to_parameter_choices.data_subsampling_number_of_samples_analysis import (
    run_data_subsampling_number_of_samples_analysis,
)
from topollm.config_classes.constants import (
    HYDRA_CONFIGS_BASE_PATH,
    NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS,
    TOPO_LLM_REPOSITORY_BASE_PATH,
)
from topollm.config_classes.setup_OmegaConf import setup_omega_conf
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.log_dataframe_info import log_dataframe_info
from topollm.logging.log_list_info import log_list_info
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig
    from topollm.path_management.embeddings.protocol import EmbeddingsPathManager

try:
    from hydra_plugins import hpc_submission_launcher

    hpc_submission_launcher.register_plugin()
except ImportError:
    pass

# logger for this file
global_logger: logging.Logger = logging.getLogger(
    name=__name__,
)
default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)

setup_exception_logging(
    logger=global_logger,
)

setup_omega_conf()


@hydra.main(
    config_path=f"{HYDRA_CONFIGS_BASE_PATH}",
    config_name="main_config",
    version_base="1.3",
)
def main(
    config: omegaconf.DictConfig,
) -> None:
    """Run the script."""
    logger: logging.Logger = global_logger
    logger.info(
        msg="Running script ...",
    )

    # # # # # # # # # # # # # # # # # # # # #
    # START Global settings

    # END Global settings
    # # # # # # # # # # # # # # # # # # # # #

    main_config: MainConfig = initialize_configuration(
        config=config,
        logger=logger,
    )
    verbosity: Verbosity = main_config.verbosity

    embeddings_path_manager: EmbeddingsPathManager = get_embeddings_path_manager(
        main_config=main_config,
        logger=logger,
    )
    data_dir: pathlib.Path = embeddings_path_manager.data_dir

    # ================================================== #
    #
    # ================================================== #

    # TODO: Use the LocalEstimatesSavingManager to load the computed local estimates

    # TODO: Implement the analysis here

    # TODO: Create analysis of twoNN measure for individual tokens under different noise distortions
    # TODO: (currently, we plan to create an extra script for the token-level analysis)

    # ================================================== #
    # Note: You can add additional analysis steps here
    # ================================================== #

    logger.info(
        msg="Running script DONE",
    )


if __name__ == "__main__":
    main()

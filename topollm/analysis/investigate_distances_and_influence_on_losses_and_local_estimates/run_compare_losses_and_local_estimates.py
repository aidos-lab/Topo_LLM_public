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
from dataclasses import dataclass
from typing import TYPE_CHECKING

import hydra
import hydra.core.hydra_config
import omegaconf

from topollm.analysis.investigate_distances_and_influence_on_losses_and_local_estimates.compute_predictions_on_hidden_states_of_local_estimates_container import (
    compute_predictions_on_hidden_states_of_local_estimates_container,
)
from topollm.analysis.investigate_distances_and_influence_on_losses_and_local_estimates.prediction_data_containers import (
    LocalEstimatesAndPredictionsContainer,
    LocalEstimatesAndPredictionsSavePathCollection,
)
from topollm.analysis.local_estimates_handling.saving.local_estimates_containers import LocalEstimatesContainer
from topollm.analysis.local_estimates_handling.saving.local_estimates_saving_manager import LocalEstimatesSavingManager
from topollm.config_classes.constants import (
    HYDRA_CONFIGS_BASE_PATH,
)
from topollm.config_classes.main_config import MainConfig
from topollm.config_classes.setup_OmegaConf import setup_omega_conf
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.model_handling.loaded_model_container import LoadedModelContainer
from topollm.model_handling.prepare_loaded_model_container import prepare_device_and_tokenizer_and_model
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.path_management.embeddings.protocol import EmbeddingsPathManager
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    import pathlib

    pass

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


@dataclass
class ComputationData:
    """Dataclass to hold the data for the computation."""

    main_config: MainConfig

    embeddings_path_manager: EmbeddingsPathManager
    local_estimates_saving_manager: LocalEstimatesSavingManager
    local_estimates_container: LocalEstimatesContainer
    loaded_model_container: LoadedModelContainer

    local_estimates_and_predictions_container: LocalEstimatesAndPredictionsContainer

    local_estimates_and_predictions_save_path_collection: LocalEstimatesAndPredictionsSavePathCollection

    # The descriptive string is used for logging to identify the computation data
    # (for example, to distinguish the base data from the comparison data)
    descriptive_string: str = ""


class ComputationManager:
    """Class to manage the computations."""

    def __init__(
        self,
        main_config: MainConfig,
        descriptive_string: str = "",
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Initialize the manager."""
        self.verbosity: Verbosity = verbosity
        self.logger: logging.Logger = logger

        embeddings_path_manager: EmbeddingsPathManager = get_embeddings_path_manager(
            main_config=main_config,
            logger=logger,
        )

        local_estimates_saving_manager: LocalEstimatesSavingManager = (
            LocalEstimatesSavingManager.from_embeddings_path_manager(
                embeddings_path_manager=embeddings_path_manager,
                verbosity=verbosity,
                logger=logger,
            )
        )

        local_estimates_container: LocalEstimatesContainer = local_estimates_saving_manager.load_local_estimates()

        loaded_model_container: LoadedModelContainer = prepare_device_and_tokenizer_and_model(
            main_config=main_config,
            verbosity=verbosity,
            logger=logger,
        )

        local_estimates_and_predictions_container: LocalEstimatesAndPredictionsContainer = (
            compute_predictions_on_hidden_states_of_local_estimates_container(
                local_estimates_container_to_analyze=local_estimates_container,
                array_truncation_size=main_config.analysis.investigate_distances.array_truncation_size,
                tokenizer=loaded_model_container.tokenizer,
                model=loaded_model_container.model,
                descriptive_string=descriptive_string,
                analysis_verbosity_level=verbosity,
                logger=logger,
            )
        )

        distances_and_influence_on_local_estimates_dir_absolute_path: pathlib.Path = (
            embeddings_path_manager.get_distances_and_influence_on_local_estimates_dir_absolute_path()
        )

        local_estimates_and_predictions_save_path_collection: LocalEstimatesAndPredictionsSavePathCollection = LocalEstimatesAndPredictionsSavePathCollection.from_base_directory(
            distances_and_influence_on_local_estimates_dir_absolute_path=distances_and_influence_on_local_estimates_dir_absolute_path,
        )
        local_estimates_and_predictions_save_path_collection.setup_directories()

        local_estimates_and_predictions_container.save_computation_data(
            local_estimates_and_predictions_save_path_collection=local_estimates_and_predictions_save_path_collection,
        )
        local_estimates_and_predictions_container.save_human_readable_predictions_logging(
            local_estimates_and_predictions_save_path_collection=local_estimates_and_predictions_save_path_collection,
        )
        local_estimates_and_predictions_container.run_full_analysis_and_save_results(
            local_estimates_and_predictions_save_path_collection=local_estimates_and_predictions_save_path_collection,
        )

        self.computation_data: ComputationData = ComputationData(
            main_config=main_config,
            embeddings_path_manager=embeddings_path_manager,
            local_estimates_saving_manager=local_estimates_saving_manager,
            local_estimates_container=local_estimates_container,
            loaded_model_container=loaded_model_container,
            local_estimates_and_predictions_container=local_estimates_and_predictions_container,
            local_estimates_and_predictions_save_path_collection=local_estimates_and_predictions_save_path_collection,
            descriptive_string=descriptive_string,
        )


class ComparisonManager:
    """Manager to compare the results of the computations."""

    def __init__(
        self,
        main_config_for_base_data: MainConfig,
        main_config_for_comparison_data: MainConfig | None = None,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Initialize the manager.

        If no comparison data is provided, the comparison data will be set to None.
        This mode can be used to only compute the base data.
        """
        self.computation_data_for_base_data: ComputationManager = ComputationManager(
            main_config=main_config_for_base_data,
            descriptive_string="base_data",
            verbosity=verbosity,
            logger=logger,
        )

        if main_config_for_comparison_data is not None:
            self.computation_data_for_comparison_data: ComputationManager | None = ComputationManager(
                main_config=main_config_for_comparison_data,
                descriptive_string="comparison_data",
                verbosity=verbosity,
                logger=logger,
            )
        else:
            self.computation_data_for_comparison_data = None

        # TODO: Implement calls to the comparisons if the comparison data is available


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

    # ================================================== #
    # Base data (for example, non-noised data)
    # ================================================== #

    main_config: MainConfig = initialize_configuration(
        config=config,
        logger=logger,
    )
    verbosity: Verbosity = main_config.verbosity

    # ================================================== #
    # Comparison data (for example, noise data)
    # ================================================== #

    if main_config.feature_flags.comparison.do_comparison_of_local_estimates:
        # We initialize a new main config for the comparison data
        main_config_for_comparison_data: MainConfig | None = main_config.model_copy(
            deep=True,
        )

        # Note:
        # The comparison data configs will be initialized with the default values in the config classes.
        # Make sure that you set all necessary values for the comparison data in the config file.
        main_config_for_comparison_data.local_estimates = main_config.comparison_data.local_estimates.model_copy(
            deep=True,
        )
        main_config_for_comparison_data.embeddings = main_config.comparison_data.embeddings.model_copy(
            deep=True,
        )
    else:
        # We set the comparison data to None if no comparison is requested.
        # The ComparisonManager will handle this case by not computing the comparison data and only the base data.
        main_config_for_comparison_data = None

    # ================================================== #
    # Computing data based on the configs
    # ================================================== #

    comparison_manager = ComparisonManager(
        main_config_for_base_data=main_config,
        main_config_for_comparison_data=main_config_for_comparison_data,
        verbosity=verbosity,
        logger=logger,
    )

    if main_config.feature_flags.comparison.do_comparison_of_local_estimates:
        logger.warning(
            msg="TODO: Comparisons calls are not implemented yet.",
        )
        # TODO: Add a call to the comparison methods (if applicable)

    # ================================================== #
    # Note: You can add additional analysis steps here
    # ================================================== #

    logger.info(
        msg="Running script DONE",
    )


if __name__ == "__main__":
    main()

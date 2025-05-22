# Copyright 2024-2025
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Matthias Ruppik (mail@ruppik.net)
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
import pathlib  # noqa: TCH003 - `pathlib` is used for more than type checking
from dataclasses import dataclass
from typing import TYPE_CHECKING

import hydra
import hydra.core.hydra_config
import omegaconf

from topollm.analysis.compare_local_estimates_and_distances_and_losses.compute_predictions_on_hidden_states_of_local_estimates_container import (
    compute_predictions_on_hidden_states_of_local_estimates_container,
)
from topollm.analysis.compare_local_estimates_and_distances_and_losses.prediction_data_containers import (
    LocalEstimatesAndPredictionsContainer,
    LocalEstimatesAndPredictionsSavePathCollection,
)
from topollm.analysis.local_estimates_handling.saving.local_estimates_containers import LocalEstimatesContainer
from topollm.analysis.local_estimates_handling.saving.local_estimates_saving_manager import LocalEstimatesSavingManager
from topollm.config_classes.constants import (
    HYDRA_CONFIGS_BASE_PATH,
)
from topollm.config_classes.main_config import MainConfig
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.model_handling.loaded_model_container import LoadedModelContainer
from topollm.model_handling.prepare_loaded_model_container import (
    prepare_device_and_tokenizer_and_model_from_main_config,
)
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.path_management.embeddings.protocol import EmbeddingsPathManager
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
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


@dataclass
class ComputationManagerDataContainers:
    """Dataclass to hold the containers for the computation."""

    local_estimates_container: LocalEstimatesContainer | None = None
    loaded_model_container: LoadedModelContainer | None = None
    local_estimates_and_predictions_container: LocalEstimatesAndPredictionsContainer | None = None
    local_estimates_and_predictions_save_path_collection: LocalEstimatesAndPredictionsSavePathCollection | None = None


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
        self.main_config: MainConfig = main_config

        # The descriptive string is used for logging to identify the computation data
        # (for example, to distinguish the base data from the comparison data)
        self.descriptive_string: str = descriptive_string

        self.verbosity: Verbosity = verbosity
        self.logger: logging.Logger = logger

        self.embeddings_path_manager: EmbeddingsPathManager = get_embeddings_path_manager(
            main_config=self.main_config,
            logger=self.logger,
        )

        self.local_estimates_saving_manager: LocalEstimatesSavingManager = (
            LocalEstimatesSavingManager.from_embeddings_path_manager(
                embeddings_path_manager=self.embeddings_path_manager,
                verbosity=self.verbosity,
                logger=self.logger,
            )
        )

        # Initialize the containers.
        # These are private attributes and will be set to None if not used,
        # use the getter methods to access them.
        self._containers = ComputationManagerDataContainers()

    def get_loaded_model_container(
        self,
    ) -> LoadedModelContainer:
        """Get the loaded model container."""
        if self._containers.loaded_model_container is None:
            self._containers.loaded_model_container = prepare_device_and_tokenizer_and_model_from_main_config(
                main_config=self.main_config,
                verbosity=self.verbosity,
                logger=self.logger,
            )

        return self._containers.loaded_model_container

    def get_local_estimates_container(
        self,
    ) -> LocalEstimatesContainer:
        """Get the local estimates container."""
        if self._containers.local_estimates_container is None:
            self._containers.local_estimates_container = self.local_estimates_saving_manager.load_local_estimates()

        return self._containers.local_estimates_container

    def get_local_estimates_and_predictions_container(
        self,
    ) -> LocalEstimatesAndPredictionsContainer:
        """Get the local estimates and predictions container."""
        if self._containers.local_estimates_and_predictions_container is None:
            self._containers.local_estimates_and_predictions_container = (
                compute_predictions_on_hidden_states_of_local_estimates_container(
                    local_estimates_container_to_analyze=self.get_local_estimates_container(),
                    array_truncation_size=self.main_config.analysis.investigate_distances.array_truncation_size,
                    tokenizer=self.get_loaded_model_container().tokenizer,
                    model=self.get_loaded_model_container().model,
                    descriptive_string=self.descriptive_string,
                    analysis_verbosity_level=self.verbosity,
                    logger=self.logger,
                )
            )

        return self._containers.local_estimates_and_predictions_container

    def get_local_estimates_and_predictions_save_path_collection(
        self,
    ) -> LocalEstimatesAndPredictionsSavePathCollection:
        """Get the local estimates and predictions save path collection."""
        if self._containers.local_estimates_and_predictions_save_path_collection is None:
            distances_and_influence_on_local_estimates_dir_absolute_path: pathlib.Path = (
                self.embeddings_path_manager.get_distances_and_influence_on_local_estimates_dir_absolute_path()
            )

            self._containers.local_estimates_and_predictions_save_path_collection = LocalEstimatesAndPredictionsSavePathCollection.from_base_directory(
                distances_and_influence_on_local_estimates_dir_absolute_path=distances_and_influence_on_local_estimates_dir_absolute_path,
            )
            self._containers.local_estimates_and_predictions_save_path_collection.setup_directories()

        return self._containers.local_estimates_and_predictions_save_path_collection

    def run_local_estimates_and_predictions_analysis_and_save_results(
        self,
    ) -> None:
        """Run the analysis and save the results."""
        self.get_local_estimates_and_predictions_container().save_computation_data(
            local_estimates_and_predictions_save_path_collection=self.get_local_estimates_and_predictions_save_path_collection(),
        )
        self.get_local_estimates_and_predictions_container().save_human_readable_predictions_logging(
            local_estimates_and_predictions_save_path_collection=self.get_local_estimates_and_predictions_save_path_collection(),
        )
        self.get_local_estimates_and_predictions_container().run_full_analysis_and_save_results(
            local_estimates_and_predictions_save_path_collection=self.get_local_estimates_and_predictions_save_path_collection(),
        )


@dataclass
class ComputationManagersContainer:
    """Container for different computation managers."""

    base_data: ComputationManager
    comparison_data: ComputationManager | None = None

    def log_info(
        self,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Log the info of the computation managers."""
        logger.info(
            msg=f"Computation manager for base data:\n"  # noqa: G004 - low overhead
            f"{self.base_data.get_local_estimates_container().get_summary_string()}",
        )
        if self.comparison_data is not None:
            logger.info(
                msg=f"Computation manager for comparison data:\n"  # noqa: G004 - low overhead
                f"{self.comparison_data.get_local_estimates_container().get_summary_string()}",
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
        self.main_config_for_base_data: MainConfig = main_config_for_base_data
        self.main_config_for_comparison_data: MainConfig | None = main_config_for_comparison_data
        self.verbosity: Verbosity = verbosity
        self.logger: logging.Logger = logger

        # Initialize the computation data for the base and comparison data.
        computation_manager_for_base_data: ComputationManager = ComputationManager(
            main_config=main_config_for_base_data,
            descriptive_string="base_data",
            verbosity=verbosity,
            logger=logger,
        )

        if main_config_for_comparison_data is not None:
            computation_manager_for_comparison_data: ComputationManager | None = ComputationManager(
                main_config=main_config_for_comparison_data,
                descriptive_string="comparison_data",
                verbosity=verbosity,
                logger=logger,
            )
        else:
            computation_manager_for_comparison_data = None

        self.computation_managers: ComputationManagersContainer = ComputationManagersContainer(
            base_data=computation_manager_for_base_data,
            comparison_data=computation_manager_for_comparison_data,
        )

    def run_comparison_of_local_estimates_with_losses(
        self,
    ) -> None:
        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg="Calling function for base data ...",
            )
        self.computation_managers.base_data.run_local_estimates_and_predictions_analysis_and_save_results()
        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg="Calling function for base data DONE",
            )

        if self.computation_managers.comparison_data is not None:
            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    msg="Calling function for comparison data ...",
                )
            self.computation_managers.comparison_data.run_local_estimates_and_predictions_analysis_and_save_results()
            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    msg="Calling function for comparison data DONE",
                )
        elif self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg="No comparison data provided, skipping loss comparison.",
            )

    def run_comparison_of_local_estimates_between_base_data_and_comparison_data(
        self,
    ) -> None:
        if self.computation_managers.comparison_data is None:
            self.logger.critical(
                msg="No comparison data provided, skipping comparison of local estimates between base and comparison data.",
            )
            return

        self.logger.critical(
            msg="Comparison of local estimates between base and comparison data is not implemented yet.",
        )

        # Note: This is where you would implement your comparison logic.


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
    # Comparison data:
    # for example, noise data or data with different parameters
    # ================================================== #

    if main_config.feature_flags.comparison.do_comparison_of_local_estimates_between_base_data_and_comparison_data:
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

    if verbosity >= Verbosity.NORMAL:
        comparison_manager.computation_managers.log_info(
            logger=logger,
        )

    # ================================================== #
    # Model predictions computation and
    # comparison between local estimates and losses.
    # ================================================== #

    if main_config.feature_flags.comparison.do_comparison_of_local_estimates_with_losses:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Running model predictions computation and comparison between local estimates and losses ...",
            )

        comparison_manager.run_comparison_of_local_estimates_with_losses()

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Running model predictions computation and comparison between local estimates and losses DONE",
            )

    # ================================================== #
    # Comparison between base data and comparison data
    # ================================================== #

    if main_config.feature_flags.comparison.do_comparison_of_local_estimates_between_base_data_and_comparison_data:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Running comparison between base data and comparison data ...",
            )

        comparison_manager.run_comparison_of_local_estimates_between_base_data_and_comparison_data()

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Running comparison between base data and comparison data DONE",
            )

    # ================================================== #
    # Note: You can add additional analysis steps here
    # ================================================== #

    logger.info(
        msg="Running script DONE",
    )


if __name__ == "__main__":
    main()

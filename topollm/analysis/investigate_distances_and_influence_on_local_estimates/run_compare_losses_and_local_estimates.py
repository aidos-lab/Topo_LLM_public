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
from dataclasses import dataclass
from typing import TYPE_CHECKING

import hydra
import hydra.core.hydra_config
import numpy as np
import omegaconf
import pandas as pd
from tqdm import tqdm
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from topollm.analysis.correlation.compute_correlations_with_count import compute_correlations_with_count
from topollm.analysis.investigate_distances_and_influence_on_local_estimates.compute_model_lm_head_predictions_on_vector import (
    compute_model_lm_head_predictions_on_vector,
)
from topollm.analysis.investigate_distances_and_influence_on_local_estimates.prediction_data_containers import (
    LMHeadPredictionResults,
    LocalEstimateAndPrediction,
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
from topollm.logging.log_dataframe_info import log_dataframe_info
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.model_handling.loaded_model_container import LoadedModelContainer
from topollm.model_handling.prepare_loaded_model_container import prepare_device_and_tokenizer_and_model
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.path_management.embeddings.protocol import EmbeddingsPathManager
from topollm.storage.saving_and_loading_functions.saving_and_loading import (
    save_dataframe_as_csv,
    save_python_dict_as_json,
)
from topollm.typing.enums import EmbeddingDataHandlerMode, Verbosity

if TYPE_CHECKING:
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


class LocalEstimatesAndPredictionsContainer:
    """Container for the results of a predictions computation."""

    def __init__(
        self,
        local_estimates_and_predictions_results_list: list[LocalEstimateAndPrediction],
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Initialize the container."""
        self.local_estimates_and_predictions_results_list: list[LocalEstimateAndPrediction] = (
            local_estimates_and_predictions_results_list
        )

        self.verbosity: Verbosity = verbosity
        self.logger: logging.Logger = logger

    def get_loss_vector(
        self,
    ) -> np.ndarray:
        """Extract the loss values from the container.

        I.e., this returns a vector of the loss/pseudo-perplexity values for each token.
        """
        loss_vector: np.ndarray = np.array(
            object=[
                local_estimate_and_prediction.lm_head_prediction_results.loss
                for local_estimate_and_prediction in self.local_estimates_and_predictions_results_list
            ],
        )

        return loss_vector

    def get_local_estimates_vector(
        self,
    ) -> np.ndarray:
        """Extract the local estimates from the container.

        I.e., this returns a vector of the local estimates for each token.
        """
        local_estimates_vector: np.ndarray = np.array(
            object=[
                local_estimate_and_prediction.extracted_local_estimate
                for local_estimate_and_prediction in self.local_estimates_and_predictions_results_list
            ],
        )

        return local_estimates_vector

    def create_descriptive_statistics(
        self,
    ) -> dict:
        """Create descriptive statistics for the local estimates and the loss values."""
        local_estimates_vector: np.ndarray = self.get_local_estimates_vector()
        loss_vector: np.ndarray = self.get_loss_vector()

        local_estimates_series: pd.Series = pd.Series(
            data=local_estimates_vector,
        )
        loss_series: pd.Series = pd.Series(
            data=loss_vector,
        )

        local_estimates_descriptive_sttistics: pd.Series = local_estimates_series.describe()
        loss_descriptive_statistics: pd.Series = loss_series.describe()

        # Note: The conversion of the pd.Series to dicts is necessary to be able to save the results as JSON.
        # Otherwise, we run into the error that the pd.Series is not JSON serializable.
        descriptive_statistics_dict: dict[str, dict] = {
            "local_estimates": local_estimates_descriptive_sttistics.to_dict(),
            "loss": loss_descriptive_statistics.to_dict(),
        }

        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg=f"descriptive_statistics_dict:\n{pprint.pformat(object=descriptive_statistics_dict)}",  # noqa: G004 - low overhead
            )

        return descriptive_statistics_dict

    def compute_correlation_between_local_estimates_and_loss_values(
        self,
    ) -> pd.DataFrame:
        """Compute the correlation between the local estimates and the loss values."""
        local_estimates_vector: np.ndarray = self.get_local_estimates_vector()
        loss_vector: np.ndarray = self.get_loss_vector()

        # Convert the local estimates and loss values to a DataFrame
        local_estimates_and_loss_df: pd.DataFrame = pd.DataFrame(
            data={
                "local_estimate": local_estimates_vector,
                "loss": loss_vector,
            },
        )

        correlations_df: pd.DataFrame = compute_correlations_with_count(
            df=local_estimates_and_loss_df,
            cols=[
                "local_estimate",
                "loss",
            ],
            methods=None,  # default correlation methods are used
            significance_level=0.05,
        )

        if self.verbosity >= Verbosity.NORMAL:
            log_dataframe_info(
                df=correlations_df,
                df_name="correlations_df",
                logger=self.logger,
            )

        return correlations_df

    def run_full_analysis_and_save_results(
        self,
        local_estimates_and_predictions_save_path_collection: LocalEstimatesAndPredictionsSavePathCollection,
    ) -> None:
        """Run function to call the different analysis steps."""
        descriptive_statistics_dict: dict = self.create_descriptive_statistics()

        save_python_dict_as_json(
            python_dict=descriptive_statistics_dict,
            save_path=local_estimates_and_predictions_save_path_collection.descriptive_statistics_dict_save_path,
            python_dict_name_for_logging="descriptive_statistics_dict",
            verbosity=self.verbosity,
            logger=self.logger,
        )

        correlations_df: pd.DataFrame = self.compute_correlation_between_local_estimates_and_loss_values()

        save_dataframe_as_csv(
            dataframe=correlations_df,
            save_path=local_estimates_and_predictions_save_path_collection.correlations_df_save_path,
            dataframe_name_for_logging="correlations_df",
            verbosity=self.verbosity,
            logger=self.logger,
        )

    # TODO: Implement methods to compare local estimates and loss values between two containers
    # TODO: Implement methods to save the comparison results to disk

    # TODO: Implement saving the results list in a human readable format to disk


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

    @staticmethod
    def from_main_config(
        main_config: MainConfig,
        descriptive_string: str = "",
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> "ComputationData":
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
            compute_predictions_on_hidden_states(
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

        local_estimates_and_predictions_container.run_full_analysis_and_save_results(
            local_estimates_and_predictions_save_path_collection=local_estimates_and_predictions_save_path_collection,
        )

        result: ComputationData = ComputationData(
            main_config=main_config,
            embeddings_path_manager=embeddings_path_manager,
            local_estimates_saving_manager=local_estimates_saving_manager,
            local_estimates_container=local_estimates_container,
            loaded_model_container=loaded_model_container,
            local_estimates_and_predictions_container=local_estimates_and_predictions_container,
            local_estimates_and_predictions_save_path_collection=local_estimates_and_predictions_save_path_collection,
            descriptive_string=descriptive_string,
        )

        return result


class ComparisonManager:
    """Manager to compare the results of the computations."""

    def __init__(
        self,
        main_config_for_base_data: MainConfig,
        main_config_for_comparison_data: MainConfig,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Initialize the manager."""
        self.computation_data_for_base_data: ComputationData = ComputationData.from_main_config(
            main_config=main_config_for_base_data,
            descriptive_string="base_data",
            verbosity=verbosity,
            logger=logger,
        )
        self.computation_data_for_comparison_data: ComputationData = ComputationData.from_main_config(
            main_config=main_config_for_comparison_data,
            descriptive_string="comparison_data",
            verbosity=verbosity,
            logger=logger,
        )

        # TODO: Implement calls to the comparisons


# # # # # # # # # # # # # # # # # # # # #
# Model inference and predictions


def compute_predictions_on_hidden_states(
    local_estimates_container_to_analyze: LocalEstimatesContainer,
    array_truncation_size: int,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    model: PreTrainedModel,
    descriptive_string: str = "",
    analysis_verbosity_level: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> LocalEstimatesAndPredictionsContainer:
    """Compute and collect the model predictions for the given hidden states.

    The descriptive string is used for logging to identify the computation data.
    """
    array_to_analyze: np.ndarray | None = local_estimates_container_to_analyze.array_for_estimator_np
    if array_to_analyze is None:
        msg = "The array_to_analyze is None."
        raise ValueError(
            msg,
        )

    corresponding_metadata: pd.DataFrame | None = local_estimates_container_to_analyze.pointwise_results_meta_frame
    if corresponding_metadata is None:
        msg = "The corresponding metadata is None."
        raise ValueError(
            msg,
        )

    pointwise_local_estimates_to_analyze: np.ndarray = local_estimates_container_to_analyze.pointwise_results_array_np

    results_list: list[LocalEstimateAndPrediction] = []

    for vector_index in tqdm(
        iterable=range(array_truncation_size),
        desc=f"Iterating over vectors for {descriptive_string = }",
    ):
        # Extract the embedding vector for the vector_index and the corresponding metadata
        extracted_vector = array_to_analyze[vector_index]
        extracted_metadata = corresponding_metadata.iloc[vector_index]

        # Note: You need to be careful that because of the potential vector de-duplication step
        # in the PreparedData creation, that the `vector_index` does not necessarily agree with the original index
        # of the token in the prepared data.
        # In particular, this might lead to issues when trying to compare embedding vectors and losses
        # between spaces that came from different embedding models or token embedding modes.
        # To help with the comparison,
        # we will save the original token index from the embeddings data preparation script.

        # Make a copy of the metadata to avoid modifying the original DataFrame
        extracted_metadata_reduced = extracted_metadata.copy()
        # Remove the "tokens_list" and "concatenated_tokens" columns from the metadata,
        # so that we do not save too much data to disk
        if "tokens_list" in extracted_metadata_reduced:
            extracted_metadata_reduced = extracted_metadata_reduced.drop(
                labels="tokens_list",
                axis=0,
            )
        if "concatenated_tokens" in extracted_metadata_reduced:
            extracted_metadata_reduced = extracted_metadata_reduced.drop(
                labels="concatenated_tokens",
                axis=0,
            )

        original_prepared_data_index = None  # TODO: .index does not give the right thing here

        # # # #
        extracted_local_estimate = pointwise_local_estimates_to_analyze[vector_index]

        if analysis_verbosity_level >= Verbosity.DEBUG:
            logger.info(
                msg=f"extracted_metadata:\n{extracted_metadata}",  # noqa: G004 - low overhead
            )
            logger.info(
                msg=f"extracted_local_estimate:\n{extracted_local_estimate}",  # noqa: G004 - low overhead
            )
            if "tokens_list" in extracted_metadata:
                logger.info(
                    msg=f"tokens_list:\n{extracted_metadata["tokens_list"]}",  # noqa: G004 - low overhead
                )

        # # # #
        # Forward pass through the model and compute predictions and loss
        lm_head_prediction_results: LMHeadPredictionResults = compute_model_lm_head_predictions_on_vector(
            tokenizer=tokenizer,
            model=model,
            extracted_vector=extracted_vector,
            extracted_metadata=extracted_metadata,
            top_k=10,
            verbosity=analysis_verbosity_level,
            logger=logger,
        )

        local_estimate_and_prediction: LocalEstimateAndPrediction = LocalEstimateAndPrediction(
            vector_index=vector_index,
            extracted_local_estimate=extracted_local_estimate,
            lm_head_prediction_results=lm_head_prediction_results,
            extracted_metadata_reduced=extracted_metadata_reduced.to_dict(),
            original_prepared_data_index=None,  # TODO: Add the original index in the prepared data to the results
        )

        results_list.append(
            local_estimate_and_prediction,
        )

    local_estimates_and_predictions_container: LocalEstimatesAndPredictionsContainer = (
        LocalEstimatesAndPredictionsContainer(
            local_estimates_and_predictions_results_list=results_list,
        )
    )

    return local_estimates_and_predictions_container


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

    main_config_for_comparison_data: MainConfig = main_config.model_copy(
        deep=True,
    )

    # TODO: Check our current decision: We do not necesarily want to initialize the comparisons configs with the defaults.
    main_config_for_comparison_data.local_estimates = main_config.comparison_data.local_estimates.model_copy(
        deep=True,
    )
    main_config_for_comparison_data.embeddings = main_config.comparison_data.embeddings.model_copy(
        deep=True,
    )

    # ================================================== #
    # Computing data based on the configs
    # ================================================== #

    comparison_manager = ComparisonManager(
        main_config_for_base_data=main_config,
        main_config_for_comparison_data=main_config_for_comparison_data,
        verbosity=verbosity,
        logger=logger,
    )

    # TODO: Create analysis of twoNN measure for individual tokens under different noise distortions

    # ================================================== #
    # Note: You can add additional analysis steps here
    # ================================================== #

    logger.info(
        msg="Running script DONE",
    )


if __name__ == "__main__":
    main()

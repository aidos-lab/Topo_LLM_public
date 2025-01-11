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

import logging
import pathlib
import pprint
from dataclasses import dataclass

import numpy as np
import pandas as pd

from topollm.analysis.correlation.compute_correlations_with_count import compute_correlations_with_count
from topollm.logging.log_dataframe_info import log_dataframe_info
from topollm.storage.saving_and_loading_functions.saving_and_loading import (
    save_dataframe_as_csv,
    save_numpy_array_as_npy,
    save_python_dict_as_json,
)
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


@dataclass
class LMHeadPredictionResults:
    """Container for the results of the LM head predictions."""

    output_logits_softmax_np: np.ndarray | None
    top_k_tokens: list[str]
    top_k_probabilities: list[float]
    loss: float | None

    actual_token_id: int | None = None
    actual_token_name: str | None = None


@dataclass
class LocalEstimatesAndPredictionsSavePathCollection:
    """Dataclass to hold the paths for saving and loading the results."""

    # This is the base directory under which all the other files will be placed
    distances_and_influence_on_losses_and_local_estimates_dir_absolute_path: pathlib.Path

    # Statistics
    descriptive_statistics_dict_save_path: pathlib.Path

    # Arrays
    loss_vector_save_path: pathlib.Path

    # Correlation
    correlations_df_save_path: pathlib.Path

    @staticmethod
    def from_base_directory(
        distances_and_influence_on_local_estimates_dir_absolute_path: pathlib.Path,
    ) -> "LocalEstimatesAndPredictionsSavePathCollection":
        """Create a new instance of the dataclass from the base directory."""
        descriptive_statistics_dict_save_path: pathlib.Path = pathlib.Path(
            distances_and_influence_on_local_estimates_dir_absolute_path,
            "descriptive_statistics_dict.json",
        )

        loss_vector_save_path: pathlib.Path = pathlib.Path(
            distances_and_influence_on_local_estimates_dir_absolute_path,
            "arrays",
            "loss_vector.np",
        )

        correlations_df_save_path: pathlib.Path = pathlib.Path(
            distances_and_influence_on_local_estimates_dir_absolute_path,
            "correlations_df.csv",
        )

        return LocalEstimatesAndPredictionsSavePathCollection(
            distances_and_influence_on_losses_and_local_estimates_dir_absolute_path=distances_and_influence_on_local_estimates_dir_absolute_path,
            loss_vector_save_path=loss_vector_save_path,
            descriptive_statistics_dict_save_path=descriptive_statistics_dict_save_path,
            correlations_df_save_path=correlations_df_save_path,
        )

    def setup_directories(
        self,
    ) -> None:
        for path in [
            self.descriptive_statistics_dict_save_path,
            self.loss_vector_save_path,
            self.distances_and_influence_on_losses_and_local_estimates_dir_absolute_path,
        ]:
            # Create the directories if they do not exist
            if not path.parent.exists():
                path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )


@dataclass
class LocalEstimateAndPrediction:
    """Container for the local estimate and the corresponding prediction results."""

    vector_index: int

    extracted_local_estimate: float
    lm_head_prediction_results: LMHeadPredictionResults

    extracted_metadata_selected_keys: dict | None = None
    original_prepared_data_index: int | None = None


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
        # # # #
        # Statistics

        descriptive_statistics_dict: dict = self.create_descriptive_statistics()

        save_python_dict_as_json(
            python_dict=descriptive_statistics_dict,
            save_path=local_estimates_and_predictions_save_path_collection.descriptive_statistics_dict_save_path,
            python_dict_name_for_logging="descriptive_statistics_dict",
            verbosity=self.verbosity,
            logger=self.logger,
        )

        # # # #
        # Arrays

        loss_vector: np.ndarray = self.get_loss_vector()

        save_numpy_array_as_npy(
            array_np=loss_vector,
            save_path=local_estimates_and_predictions_save_path_collection.loss_vector_save_path,
            array_name_for_logging="loss_vector",
            verbosity=self.verbosity,
            logger=self.logger,
        )

        # # # #
        # Correlation analysis

        correlations_df: pd.DataFrame = self.compute_correlation_between_local_estimates_and_loss_values()

        save_dataframe_as_csv(
            dataframe=correlations_df,
            save_path=local_estimates_and_predictions_save_path_collection.correlations_df_save_path,
            dataframe_name_for_logging="correlations_df",
            verbosity=self.verbosity,
            logger=self.logger,
        )

        # TODO: Implement saving the results list in a human readable format to disk

    # TODO: Implement methods to compare local estimates and loss values between two containers
    # TODO: Implement methods to save the comparison results to disk

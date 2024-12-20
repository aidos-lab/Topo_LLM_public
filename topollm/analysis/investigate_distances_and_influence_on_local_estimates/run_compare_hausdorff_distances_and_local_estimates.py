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
from dataclasses import dataclass
from typing import TYPE_CHECKING

import hydra
import hydra.core.hydra_config
import joblib
import numpy as np
import omegaconf
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from topollm.analysis.correlation.compute_correlations_with_count import compute_correlations_with_count
from topollm.analysis.local_estimates_handling.saving.local_estimates_containers import LocalEstimatesContainer
from topollm.analysis.local_estimates_handling.saving.local_estimates_saving_manager import LocalEstimatesSavingManager
from topollm.config_classes.constants import (
    HYDRA_CONFIGS_BASE_PATH,
    NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS,
    TOPO_LLM_REPOSITORY_BASE_PATH,
)
from topollm.config_classes.main_config import MainConfig
from topollm.config_classes.setup_OmegaConf import setup_omega_conf
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.log_dataframe_info import log_dataframe_info
from topollm.logging.log_list_info import log_list_info
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.model_handling.loaded_model_container import LoadedModelContainer
from topollm.model_handling.prepare_loaded_model_container import prepare_device_and_tokenizer_and_model
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.typing.enums import ArtificialNoiseMode, Verbosity

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
    # TODO: Check that the global settings are correct for the current analysis

    array_truncation_size: int = 5_000
    # array_truncation_size: int = 500

    # NOTE: We will probably implement the following two analysis steps in a separate script which iterates over the directory structure
    # TODO(Ben): Implement iteration over different noise levels and noise seeds, to make a plot of Hausdorff distances vs. local estimates for each noise level and seed.
    # TODO(Ben): Plot of Hausdorff distances vs. global estimates.

    # END Global settings
    # # # # # # # # # # # # # # # # # # # # #

    main_config: MainConfig = initialize_configuration(
        config=config,
        logger=logger,
    )
    verbosity: Verbosity = main_config.verbosity

    embeddings_path_manager_for_base_data: EmbeddingsPathManager = get_embeddings_path_manager(
        main_config=main_config,
        logger=logger,
    )
    data_dir: pathlib.Path = embeddings_path_manager_for_base_data.data_dir

    # ================================================== #
    # Base data (for example, non-noised data)
    # ================================================== #

    local_estimates_saving_manager_for_base_data = LocalEstimatesSavingManager(
        embeddings_path_manager=embeddings_path_manager_for_base_data,
        verbosity=verbosity,
        logger=logger,
    )

    local_estimates_container_base_data: LocalEstimatesContainer = (
        local_estimates_saving_manager_for_base_data.load_local_estimates()
    )

    # ================================================== #
    # Comparison data (for example, noise data)
    # ================================================== #

    # TODO: We currently set the comparison data manually in this script
    artificial_noise_mode = ArtificialNoiseMode.GAUSSIAN
    artificial_noise_distortion_parameter = 0.01
    artificial_noise_seed = 4

    main_config_for_comparison_data: MainConfig = main_config.model_copy(
        deep=True,
    )
    main_config_for_comparison_data.local_estimates.noise.artificial_noise_mode = artificial_noise_mode
    main_config_for_comparison_data.local_estimates.noise.distortion_parameter = artificial_noise_distortion_parameter
    main_config_for_comparison_data.local_estimates.noise.seed = artificial_noise_seed

    embeddings_path_manager_for_comparison_data: EmbeddingsPathManager = get_embeddings_path_manager(
        main_config=main_config_for_comparison_data,
        logger=logger,
    )

    local_estimates_saving_manager_for_comparison_data = LocalEstimatesSavingManager(
        embeddings_path_manager=embeddings_path_manager_for_comparison_data,
        verbosity=verbosity,
        logger=logger,
    )

    local_estimates_container_for_comparison_data: LocalEstimatesContainer = (
        local_estimates_saving_manager_for_comparison_data.load_local_estimates()
    )

    # # # #
    # Example: Accessing the arrays used for the local estimates

    array_for_base_data: np.ndarray | None = local_estimates_container_base_data.array_for_estimator_np
    if array_for_base_data is None:
        msg = "The array for the estimator is None."
        raise ValueError(
            msg,
        )

    array_for_comparison_data: np.ndarray | None = local_estimates_container_for_comparison_data.array_for_estimator_np
    if array_for_comparison_data is None:
        msg = "The array for the estimator is None."
        raise ValueError(
            msg,
        )

    # ================================================== #
    # Computing model predictions
    # ================================================== #

    loaded_model_container_for_base_data: LoadedModelContainer = prepare_device_and_tokenizer_and_model(
        main_config=main_config,
        verbosity=verbosity,
        logger=logger,
    )

    tokenizer_for_base_data: PreTrainedTokenizer | PreTrainedTokenizerFast = (
        loaded_model_container_for_base_data.tokenizer
    )
    model_for_base_data: PreTrainedModel = loaded_model_container_for_base_data.model

    results_container_for_base_data: LocalEstimatesAndPredictionsContainer = compute_predictions_on_hidden_states(
        local_estimates_container_to_analyze=local_estimates_container_base_data,
        array_truncation_size=array_truncation_size,
        tokenizer_for_base_data=tokenizer_for_base_data,
        model_for_base_data=model_for_base_data,
        analysis_verbosity_level=verbosity,
        logger=logger,
    )

    results_container_for_base_data.run_full_analysis()

    # Note that we use the same tokenizer and model for inference on the comparison data
    results_container_for_comparison_data: LocalEstimatesAndPredictionsContainer = compute_predictions_on_hidden_states(
        local_estimates_container_to_analyze=local_estimates_container_for_comparison_data,
        array_truncation_size=array_truncation_size,
        tokenizer_for_base_data=tokenizer_for_base_data,
        model_for_base_data=model_for_base_data,
        analysis_verbosity_level=verbosity,
        logger=logger,
    )

    # TODO: Call the analysis of the individual results containers

    # TODO: Compare the results for the base data and the comparison data

    # TODO: Create analysis of twoNN measure for individual tokens under different noise distortions

    # ================================================== #
    # Note: You can add additional analysis steps here
    # ================================================== #

    logger.info(
        msg="Running script DONE",
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
class LocalEstimateAndPrediction:
    """Container for the local estimate and the corresponding prediction results."""

    vector_index: int

    extracted_local_estimate: float
    lm_head_prediction_results: LMHeadPredictionResults


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

        descriptive_statistics_dict: dict = {
            "local_estimates": local_estimates_descriptive_sttistics,
            "loss": loss_descriptive_statistics,
        }

        return descriptive_statistics_dict

    def save_statistics():
        # TODO: Implement saving the descriptive statistics to disk
        pass

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
            cols=["local_estimate", "loss"],
            methods=None,  # default correlation methods are used
            significance_level=0.05,
        )

        return correlations_df

        # TODO: Compute the p-values for the correlation coefficients

    def run_full_analysis(
        self,
    ):
        """Run function to call the different analysis steps."""
        correlations_df: pd.DataFrame = self.compute_correlation_between_local_estimates_and_loss_values()

        if self.verbosity >= Verbosity.NORMAL:
            log_dataframe_info(
                df=correlations_df,
                df_name="correlations_df",
                logger=self.logger,
            )
        # TODO: Save the correlations to disk

        # TODO: Make descriptive statistics
        # TODO: Save descriptive statistics

    # TODO: Implement method for computing correlation between local estimates and loss values
    # TODO: Implement methods to compare local estimates and loss values between two containers

    # TODO: Implement saving the results list in a human readable format to disk


def compute_predictions_on_hidden_states(
    local_estimates_container_to_analyze: LocalEstimatesContainer,
    array_truncation_size: int,
    tokenizer_for_base_data: PreTrainedTokenizer | PreTrainedTokenizerFast,
    model_for_base_data: PreTrainedModel,
    analysis_verbosity_level: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> LocalEstimatesAndPredictionsContainer:
    """Compute and collect the model predictions for the given hidden states."""
    array_to_analyze = local_estimates_container_to_analyze.array_for_estimator_np
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
        desc="Iterating over vectors",
    ):
        # Extract the embedding vector for the vector_index and the corresponding metadata
        extracted_vector = array_to_analyze[vector_index]
        extracted_metadata = corresponding_metadata.iloc[vector_index]

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
            tokenizer=tokenizer_for_base_data,
            model=model_for_base_data,
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


def compute_model_lm_head_predictions_on_vector(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    model: PreTrainedModel,
    extracted_vector: np.ndarray,
    extracted_metadata: pd.Series | None,
    top_k: int = 10,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> LMHeadPredictionResults:
    """Compute the model predictions for the given vector and metadata."""
    # Convert the NumPy array to a PyTorch tensor with shape [batch_size, sequence_length, hidden_size]
    # In this case, batch_size=1, sequence_length=1
    extracted_tensor: torch.Tensor = (
        torch.tensor(
            data=extracted_vector,
            dtype=torch.float32,
        )
        .unsqueeze(dim=0)
        .unsqueeze(dim=0)
    )
    # Move to the device
    extracted_tensor = extracted_tensor.to(
        device=model.device,
    )

    # Forward pass through the language model head
    # Shape: [batch_size=1, sequence_length=1, vocab_size]
    output_logits = model.lm_head(
        extracted_tensor,
    )

    output_logits_softmax: torch.Tensor = torch.softmax(
        input=output_logits,
        dim=-1,
    ).to(
        device="cpu",
    )

    # top-K predictions
    (
        top_k_probs,
        top_k_indices,
    ) = torch.topk(
        input=output_logits_softmax,
        k=top_k,
        dim=-1,
    )

    # Decode the top-K predictions
    top_k_tokens_wrapped: list = [tokenizer.convert_ids_to_tokens([idx]) for idx in top_k_indices[0, 0].tolist()]
    top_k_tokens: list[str] = [token_in_list[0] for token_in_list in top_k_tokens_wrapped]
    top_k_probabilities: list[float] = top_k_probs[0, 0].tolist()

    # Log the top-K predictions and their probabilities
    if verbosity >= Verbosity.DEBUG:
        if extracted_metadata is not None:
            logger.info(
                msg=f"token_id:\n{extracted_metadata['token_id']}",  # noqa: G004 - low overhead
            )
            logger.info(
                msg=f"token_name:\n{extracted_metadata['token_name']}",  # noqa: G004 - low overhead
            )
        logger.info(
            msg=f"Top-K predicted tokens:\n{top_k_tokens}",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"Top-K probabilities:\n{top_k_probabilities}",  # noqa: G004 - low overhead
        )

    # # # #
    # Compute the masked language model loss if the metadata is available
    if extracted_metadata is not None and "token_id" in extracted_metadata and "token_name" in extracted_metadata:
        actual_token_id = extracted_metadata["token_id"]
        actual_token_name = extracted_metadata["token_name"]

        # Reshape prediction_scores and actual_token_id for the loss computation
        loss = compute_masked_language_model_loss(
            actual_token_id=actual_token_id,
            output_logits=output_logits,
            model=model,
            verbosity=verbosity,
            logger=logger,
        )
        loss_value = float(loss.item())
    else:
        actual_token_id = None
        actual_token_name = None
        loss_value = None

    lm_head_prediction_results: LMHeadPredictionResults = LMHeadPredictionResults(
        output_logits_softmax_np=output_logits_softmax.detach().cpu().numpy(),
        top_k_tokens=top_k_tokens,
        top_k_probabilities=top_k_probabilities,
        loss=loss_value,
        actual_token_id=actual_token_id,
        actual_token_name=actual_token_name,
    )

    return lm_head_prediction_results


def compute_masked_language_model_loss(
    actual_token_id: int,
    output_logits: torch.Tensor,
    model: PreTrainedModel,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> torch.Tensor:
    """Compute the model loss, i.e., the token pseudo-perplexity."""
    loss_fct = CrossEntropyLoss()

    actual_token_id_tensor: torch.Tensor = torch.tensor(
        data=actual_token_id,
        dtype=torch.long,
    ).to(
        device=output_logits.device,
    )
    output_logits_reshaped = output_logits.view(
        -1,
        model.config.vocab_size,
    )  # Shape: [1, vocab_size]
    actual_token_id_reshaped: torch.Tensor = actual_token_id_tensor.view(
        -1,
    )  # Shape: [1]

    # Compute masked language modeling loss
    masked_lm_loss = loss_fct(
        output_logits_reshaped,
        actual_token_id_reshaped,
    )

    masked_lm_loss.to(
        device="cpu",
    )

    if verbosity >= Verbosity.DEBUG:
        logger.info(
            msg=f"masked_lm_loss:\n{masked_lm_loss.item()}",  # noqa: G004 - low overhead
        )

    return masked_lm_loss


def pairwise_distances(
    X,
) -> np.ndarray:
    """Calculate pairwise distance matrix of a given data matrix and return said matrix."""
    D = np.sum((X[None, :] - X[:, None]) ** 2, -1) ** 0.5
    return D


def get_neighbours_and_ranks(
    X,
    k,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the neighbourhoods and the ranks of a given space `X`, and return the corresponding tuple.

    An additional parameter $k$,
    the size of the neighbourhood, is required.
    """
    X = pairwise_distances(X)

    # Warning: this is only the ordering of neighbours that we need to
    # extract neighbourhoods below. The ranking comes later!
    X_ranks = np.argsort(X, axis=-1, kind="stable")

    # Extract neighbourhoods.
    X_neighbourhood = X_ranks[:, 1 : k + 1]

    # Convert this into ranks (finally)
    X_ranks = X_ranks.argsort(axis=-1, kind="stable")

    return X_neighbourhood, X_ranks


def MRRE_pointwise(
    X,
    Z,
    k,
) -> np.ndarray:
    """Calculate the pointwise mean rank distortion for each data point in the data space `X` with respect to the latent space `Z`.

    Inputs:
        - X: array of shape (m, n) (data space)
        - Z: array of shape (m, l) (latent space)
        - k: number of nearest neighbors to consider
    Output:
        - mean_rank_distortions: array of length m
    """
    X_neighbourhood, X_ranks = get_neighbours_and_ranks(X, k)
    Z_neighbourhood, Z_ranks = get_neighbours_and_ranks(Z, k)

    n = X.shape[0]
    mean_rank_distortions = np.zeros(n)

    for row in range(n):
        rank_differences = []
        for neighbour in Z_neighbourhood[row]:
            rx = X_ranks[row, neighbour]
            rz = Z_ranks[row, neighbour]
            rank_differences.append(abs(rx - rz) / rz)
        mean_rank_distortions[row] = np.mean(rank_differences)

    return mean_rank_distortions


if __name__ == "__main__":
    main()

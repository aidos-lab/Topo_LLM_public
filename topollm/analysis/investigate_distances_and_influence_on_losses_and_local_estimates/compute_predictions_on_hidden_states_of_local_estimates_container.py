import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from topollm.analysis.investigate_distances_and_influence_on_losses_and_local_estimates.compute_model_lm_head_predictions_on_vector import (
    compute_model_lm_head_predictions_on_vector,
)
from topollm.analysis.investigate_distances_and_influence_on_losses_and_local_estimates.prediction_data_containers import (
    LMHeadPredictionResults,
    LocalEstimateAndPrediction,
    LocalEstimatesAndPredictionsContainer,
)
from topollm.analysis.local_estimates_handling.saving.local_estimates_containers import LocalEstimatesContainer
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)

# # # # # # # # # # # # # # # # # # # # #
# Model inference and predictions


def compute_predictions_on_hidden_states_of_local_estimates_container(
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

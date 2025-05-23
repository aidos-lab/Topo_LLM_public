# Copyright 2024-2025
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_1 (author1@example.com)
# AUTHOR_2 (author2@example.com)
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

"""Compute predictions on hidden states of a local estimates container."""

import logging
from typing import TYPE_CHECKING

from tqdm import tqdm
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from topollm.analysis.compare_local_estimates_and_distances_and_losses.compute_model_lm_head_predictions_on_vector import (
    compute_model_lm_head_predictions_on_vector,
)
from topollm.analysis.compare_local_estimates_and_distances_and_losses.prediction_data_containers import (
    LMHeadPredictionResults,
    LocalEstimateAndPrediction,
    LocalEstimatesAndPredictionsContainer,
)
from topollm.analysis.local_estimates_handling.saving.local_estimates_containers import LocalEstimatesContainer
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

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
    *,
    drop_keys_from_metadata: list[str] | None = None,
    analysis_verbosity_level: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> LocalEstimatesAndPredictionsContainer:
    """Compute and collect the model predictions for the given hidden states.

    The descriptive string is used for logging to identify the computation data.
    """
    if drop_keys_from_metadata is None:
        # Default keys to drop from the metadata
        drop_keys_from_metadata = [
            "tokens_list",
        ]
        # Note:
        # You might want to drop other keys from the metadata,
        # such as "concatenated_tokens",
        # to avoid saving too much data to disk.

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
        extracted_metadata: pd.Series = corresponding_metadata.iloc[vector_index]

        # Note: You need to be careful that because of the potential vector de-duplication step
        # in the PreparedData creation, that the `vector_index` does not necessarily agree with the original index
        # of the token in the prepared data.
        # In particular, this might lead to issues when trying to compare embedding vectors and losses
        # between spaces that came from different embedding models or token embedding modes.
        # To help with the comparison,
        # we will save the original token index from the embeddings data preparation script.
        #
        original_prepared_data_index = corresponding_metadata.index[vector_index]

        # Make a copy of the metadata to avoid modifying the original DataFrame
        extracted_metadata_selected_keys: pd.Series = extracted_metadata.copy()
        # Remove certain columns from the metadata,
        # so that we do not save too much data to disk
        for key_to_drop in drop_keys_from_metadata:
            if key_to_drop in extracted_metadata_selected_keys:
                extracted_metadata_selected_keys = extracted_metadata_selected_keys.drop(
                    labels=key_to_drop,
                    axis=0,
                )

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
                    msg=f"tokens_list:\n{extracted_metadata['tokens_list']}",  # noqa: G004 - low overhead
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
            extracted_metadata_selected_keys=extracted_metadata_selected_keys.to_dict(),
            original_prepared_data_index=original_prepared_data_index,
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

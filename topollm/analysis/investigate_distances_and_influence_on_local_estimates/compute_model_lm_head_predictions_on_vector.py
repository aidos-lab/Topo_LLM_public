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

import numpy as np
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from topollm.analysis.investigate_distances_and_influence_on_local_estimates.prediction_data_containers import (
    LMHeadPredictionResults,
)
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


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

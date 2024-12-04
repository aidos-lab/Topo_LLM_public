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

"""Data Handler for computing and storing token-level embeddings."""

import logging

import numpy as np
import torch
import torch.utils.data
import transformers.modeling_outputs

from topollm.compute_embeddings.embedding_data_handler.base_embedding_data_handler import BaseEmbeddingDataHandler
from topollm.model_inference.perplexity.repeat_tensor_input_and_apply_diagonal_mask import (
    create_diagonal_mask_without_special_tokens,
    repeat_tensor_input_and_apply_diagonal_mask,
)

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


class MLMMaskedTokenEmbeddingDataHandler(BaseEmbeddingDataHandler):
    """Data handler for regular computation and storing of token-level embeddings."""

    def compute_embeddings_from_batch(
        self,
        batch: dict,
    ) -> np.ndarray:
        """Compute model outputs and extract embeddings from a batch."""
        inputs: dict[
            str,
            torch.Tensor,
        ] = self.prepare_model_inputs_from_batch(
            batch=batch,
        )

        mask_token_id: int | None = self.tokenizer.mask_token_id
        if mask_token_id is None:
            msg = "The tokenizer does not have a mask token."
            raise ValueError(
                msg,
            )

        # Container to hold the embedding arrays for each sequence in the input
        sequence_embedding_arrays_list = []

        # Iterate over the individual input sequences, and process each one separately.
        for single_sequence_input_ids, single_sequence_attention_mask in zip(
            inputs["input_ids"],
            inputs["attention_mask"],
            strict=True,
        ):
            # We restrict to the non-padding tokens to cut down on the total computation cost.
            single_sequence_input_ids_non_padding: torch.Tensor = single_sequence_input_ids[
                single_sequence_attention_mask == 1,
            ]

            (
                repeated_masked_input,
                _,
            ) = repeat_tensor_input_and_apply_diagonal_mask(
                tensor_input=single_sequence_input_ids_non_padding,
                mask_token_id=mask_token_id,
                device=self.device,
            )
            # `repeated_masked_input.shape = torch.Size([single_sequence_length - 2, single_sequence_length])`

            # Compute embeddings.
            # The `output_hidden_states` argument needs to be set to `True`
            # so that we can access the hidden states from the different layers.
            #
            # Note: We currently do not pass any attention masks.
            with torch.no_grad():
                repeated_masked_model_outputs: transformers.modeling_outputs.BaseModelOutput = self.model(
                    repeated_masked_input,
                    output_hidden_states=True,
                )

                # Extract embeddings of all input vectors.
                # We extract the masked tokens later.
                repeated_masked_embeddings: np.ndarray = self.embedding_extractor.extract_embeddings_from_model_outputs(
                    model_outputs=repeated_masked_model_outputs,
                )
                # `repeated_masked_embeddings.shape = (single_sequence_length - 2, single_sequence_length, embedding_dimension)`,
                # where the `- 2` results from the start and end tokens.
                # E.g., for an input sequence of 7 tokens this will be of shape `(5, 7, 768)`.

                # Note that the embeddings of the special start and end tokens are not the same for the differently masked sequences
                # in the `repeated_masked_embeddings`, since a different token was masked in each row.
                # We take the average of the different special tokens to derive a common embedding for this sequence.
                first_vectors_in_each_sequence: np.ndarray = repeated_masked_embeddings[:, 0, :]
                start_token_average_embedding = np.mean(
                    first_vectors_in_each_sequence,
                    axis=0,
                )

                last_vector_in_each_sequence: np.ndarray = repeated_masked_embeddings[:, -1, :]
                end_token_average_embedding = np.mean(
                    last_vector_in_each_sequence,
                    axis=0,
                )

                # Extract the embeddings of the masked tokens.
                diagonal_mask = create_diagonal_mask_without_special_tokens(
                    sequence_length=len(single_sequence_input_ids_non_padding),
                    device=torch.device(device="cpu"),  # We want this to be on the CPU for the numpy operations.
                )
                diagonal_mask_np: np.ndarray = diagonal_mask.cpu().numpy()
                mask_indices = np.where(diagonal_mask == 1)

                extracted_masked_embeddings: np.ndarray = repeated_masked_embeddings[mask_indices]

                # single_sequence_embeddings_with_averaged_special_tokens =

                # TODO(Ben): Extract the embedding vectors of the first special token, the "diagonal" masked tokens, the last special token and assemble into single array.

                pass  # noqa: PIE790 - this is here for setting breakpoints

        # TODO(Ben): Assemble extracted embeddings into array for the entire batch (note that we need to fill up to the padding length again).

        raise NotImplementedError

        return repeated_masked_embeddings

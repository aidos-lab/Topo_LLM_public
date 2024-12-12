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
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.utils.data
from transformers import PreTrainedModel

from topollm.compute_embeddings.embedding_data_handler.base_embedding_data_handler import BaseEmbeddingDataHandler
from topollm.compute_embeddings.embedding_extractor.protocol import EmbeddingExtractor
from topollm.model_inference.perplexity.repeat_tensor_input_and_apply_diagonal_mask import (
    create_diagonal_mask_without_special_tokens,
    repeat_tensor_input_and_apply_diagonal_mask,
)

if TYPE_CHECKING:
    import transformers.modeling_outputs

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)
default_device: torch.device = torch.device(
    device="cpu",
)


class SingleSequenceRepeatedMaskedInputBatchProcessor:
    """Batch processor for repeated masked input sequences."""

    def __init__(
        self,
        model: PreTrainedModel,
        embedding_extractor: EmbeddingExtractor,
        device: torch.device = default_device,
    ) -> None:
        """Initialize the BatchProcessor.

        Args:
            model: The pre-trained model for the forward pass.
            embedding_extractor: A helper class or function to extract embeddings.
            device: The computation device (e.g., 'cpu' or 'cuda').

        """
        self.model: PreTrainedModel = model
        self.embedding_extractor: EmbeddingExtractor = embedding_extractor
        self.device = device

    def process_in_batches(
        self,
        repeated_masked_input: torch.Tensor,
        batch_size: int,
    ) -> np.ndarray:
        """Process the repeated masked input in batches to avoid out-of-memory issues.

        Args:
            repeated_masked_input: The input tensor with shape [N, L].
            batch_size: The number of sequences to process per batch.

        Returns:
            A numpy array of shape (N, L, embedding_dim) containing the extracted embeddings.

        """
        num_sequences, sequence_length = repeated_masked_input.shape

        # Placeholder for results
        all_embeddings: list = []

        for start_idx in range(
            0,
            num_sequences,
            batch_size,
        ):
            end_idx = min(
                start_idx + batch_size,
                num_sequences,
            )

            # Slice the batch
            batch_input: torch.Tensor = repeated_masked_input[start_idx:end_idx].to(
                device=self.device,
            )

            with torch.no_grad():
                # Forward pass through the model.
                # The `output_hidden_states` argument needs to be set to `True`
                # so that we can access the hidden states from the different layers.
                #
                # Note: We currently do not pass any attention masks.
                model_outputs: transformers.modeling_outputs.BaseModelOutput = self.model(
                    batch_input,
                    output_hidden_states=True,
                )

            # Extract embeddings for the current batch
            batch_embeddings: np.ndarray = self.embedding_extractor.extract_embeddings_from_model_outputs(
                model_outputs=model_outputs,
            )

            # Accumulate embeddings
            all_embeddings.append(batch_embeddings)

        # Concatenate all batch embeddings along the sequence dimension
        return np.concatenate(
            all_embeddings,
            axis=0,
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
        list_of_sequence_embedding_arrays: list[np.ndarray] = []

        single_sequence_batch_processor = SingleSequenceRepeatedMaskedInputBatchProcessor(
            model=self.model,
            embedding_extractor=self.embedding_extractor,
            device=self.device,
        )

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

            # `repeated_masked_input.shape = torch.Size([single_sequence_length - 2, single_sequence_length])`
            (
                repeated_masked_input,
                _,
            ) = repeat_tensor_input_and_apply_diagonal_mask(
                tensor_input=single_sequence_input_ids_non_padding,
                mask_token_id=mask_token_id,
                device=self.device,
            )

            # Compute embeddings.
            # Extract embeddings of all input vectors.
            # We extract each sequence's masked token later.
            #
            # `repeated_masked_embeddings.shape =
            # (single_sequence_length - 2, single_sequence_length, embedding_dimension)`,
            # where the `- 2` results from the start and end tokens.
            # E.g., for an input sequence of 7 tokens this will be of shape `(5, 7, 768)`.
            repeated_masked_embeddings: np.ndarray = single_sequence_batch_processor.process_in_batches(
                repeated_masked_input=repeated_masked_input,
                batch_size=self.single_sequence_batch_size,
            )

            single_sequence_embeddings_with_averaged_special_tokens: np.ndarray = (
                self.extract_masked_token_embeddings_and_averaged_special_tokens(
                    repeated_masked_embeddings=repeated_masked_embeddings,
                    sequence_length=len(single_sequence_input_ids_non_padding),
                )
            )

            # Append the embeddings of the current sequence to the list.
            list_of_sequence_embedding_arrays.append(
                single_sequence_embeddings_with_averaged_special_tokens,
            )

        # # # #
        # After iterating over all sequences in the batch, we concatenate the embeddings into a single array.
        final_array: np.ndarray = self.make_final_array_from_list_of_sequence_embedding_arrays(
            inputs=inputs,
            list_of_sequence_embedding_arrays=list_of_sequence_embedding_arrays,
        )

        return final_array

    @staticmethod
    def make_final_array_from_list_of_sequence_embedding_arrays(
        inputs: dict,
        list_of_sequence_embedding_arrays: list[np.ndarray],
    ) -> np.ndarray:
        # Check that the number of computed embeddings matches the number of input sequences.
        # `len(list_of_sequence_embedding_arrays) = batch_size`
        if len(list_of_sequence_embedding_arrays) != inputs["input_ids"].shape[0]:
            msg = "The number of computed embeddings does not match the number of input sequences."
            raise ValueError(
                msg,
            )

        # Fill up the embeddings to the padding length.
        expected_sequence_length: int = inputs["input_ids"].shape[1]

        padded_sequences: list[np.ndarray] = []
        for single_sequence_embedding_array in list_of_sequence_embedding_arrays:
            padding: np.ndarray = np.zeros(
                shape=(
                    expected_sequence_length - single_sequence_embedding_array.shape[0],
                    single_sequence_embedding_array.shape[1],
                ),
            )
            padded_single_sequence_embedding_array: np.ndarray = np.vstack(
                tup=[
                    single_sequence_embedding_array,
                    padding,
                ],
            )  # Stack original sequence and zero-padding
            padded_sequences.append(
                padded_single_sequence_embedding_array,
            )

        # Stack all padded sequences to create a single array.
        # `final_array.shape = (batch_size, expected_sequence_length, embedding_dimension)`
        final_array: np.ndarray = np.stack(
            arrays=padded_sequences,
        )

        # Check that the final array has the expected shape.
        if final_array.shape[0] != inputs["input_ids"].shape[0]:
            msg = "The number of sequences in the final array does not match the number of input sequences."
            raise ValueError(
                msg,
            )
        if final_array.shape[1] != expected_sequence_length:
            msg = "The sequence length in the final array does not match the expected sequence length."
            raise ValueError(
                msg,
            )

        return final_array

    @staticmethod
    def extract_masked_token_embeddings_and_averaged_special_tokens(
        repeated_masked_embeddings: np.ndarray,
        sequence_length: int,
    ) -> np.ndarray:
        """Extract the embeddings of the masked tokens and average the embeddings of the special tokens."""
        # Note that the embeddings of the special start and end tokens
        # are not the same for the differently masked sequences
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
        diagonal_mask: torch.Tensor = create_diagonal_mask_without_special_tokens(
            sequence_length=sequence_length,
            device=torch.device(device="cpu"),  # We want this to be on the CPU for the numpy operations.
        )
        diagonal_mask_np: np.ndarray = diagonal_mask.cpu().numpy()
        mask_indices: tuple = np.where(diagonal_mask_np == 1)

        # `extracted_masked_embeddings.shape
        # = (single_sequence_length - 2, embedding_dimension)`
        extracted_masked_embeddings: np.ndarray = repeated_masked_embeddings[mask_indices]

        # Assemble the embeddings of the special tokens and the masked tokens into a single array.
        # `single_sequence_embeddings_with_averaged_special_tokens.shape =
        # (single_sequence_length, embedding_dimension)`
        single_sequence_embeddings_with_averaged_special_tokens: np.ndarray = np.vstack(
            tup=[
                start_token_average_embedding,  # Add as the first row
                extracted_masked_embeddings,  # Existing array
                end_token_average_embedding,  # Add as the last row
            ],
        )

        return single_sequence_embeddings_with_averaged_special_tokens

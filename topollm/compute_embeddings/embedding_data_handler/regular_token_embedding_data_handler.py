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

"""Data Handler for computing and storing token-level embeddings."""

import logging

import numpy as np
import torch
import torch.utils.data
import transformers.modeling_outputs

from topollm.compute_embeddings.embedding_data_handler.base_embedding_data_handler import BaseEmbeddingDataHandler

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


class RegularTokenEmbeddingDataHandler(BaseEmbeddingDataHandler):
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
        model_outputs: transformers.modeling_outputs.BaseModelOutput = self.compute_model_outputs_from_single_inputs(
            inputs=inputs,
        )
        embeddings: np.ndarray = self.embedding_extractor.extract_embeddings_from_model_outputs(
            model_outputs=model_outputs,
        )
        # `embeddings.shape = (batch_size, sequence_length, embedding_dimension)`, e.g.,
        # `embeddings.shape = (32, 512, 768)`

        return embeddings

    def compute_model_outputs_from_single_inputs(
        self,
        inputs: dict,
    ) -> transformers.modeling_outputs.BaseModelOutput:
        """Compute embeddings for the given inputs using the given model."""
        with torch.no_grad():
            # Compute embeddings.
            # The `output_hidden_states` argument needs to be set to `True`
            # so that we can access the hidden states from the different layers
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
            )

        return outputs

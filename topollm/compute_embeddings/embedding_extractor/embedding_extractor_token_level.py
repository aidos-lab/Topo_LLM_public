# Copyright 2024
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

from typing import TYPE_CHECKING

import numpy as np
import transformers.modeling_outputs

from topollm.compute_embeddings.embedding_extractor.layer_aggregator.protocol import (
    LayerAggregator,
)
from topollm.compute_embeddings.embedding_extractor.layer_extractor.protocol import (
    LayerExtractor,
)

if TYPE_CHECKING:
    import torch


class EmbeddingExtractorTokenLevel:
    """Implementation of the EmbeddingExtractor protocol which extracts token level embeddings."""

    def __init__(
        self,
        layer_extractor: LayerExtractor,
        layer_aggregator: LayerAggregator,
        embedding_dimension: int,
    ) -> None:
        """Initialize the EmbeddingExtractorTokenLevel."""
        self.layer_extractor: LayerExtractor = layer_extractor
        self.layer_aggregator: LayerAggregator = layer_aggregator
        self.embedding_dimension: int = embedding_dimension

    def extract_embeddings_from_model_outputs(
        self,
        model_outputs: transformers.modeling_outputs.BaseModelOutput,
    ) -> np.ndarray:
        # Ensure the model outputs hidden states
        if not hasattr(
            model_outputs,
            "hidden_states",
        ):
            msg = "Model outputs do not contain 'hidden_states'"
            raise ValueError(msg)

        hidden_states = model_outputs.hidden_states  # Assuming this is a tuple of tensors

        if hidden_states is None:
            msg = "'hidden_states' is None. Did you call the model with 'output_hidden_states=True'?"
            raise ValueError(msg)

        # Extract specified layers
        layers_to_extract: list[torch.Tensor] = self.layer_extractor.extract_layers_from_model_outputs(
            hidden_states=hidden_states,
        )

        # Aggregate the extracted layers
        embeddings: torch.Tensor = self.layer_aggregator.aggregate_layers(
            layers_to_extract=layers_to_extract,
        )

        return embeddings.cpu().numpy()

# coding=utf-8
#
# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Julius von Rohrscheidt (julius.rohrscheidt@helmholtz-muenchen.de)
# Benjamin Ruppik (ruppik@hhu.de)
#
# This code was generated with the help of AI writing assistants
# including GitHub Copilot, ChatGPT, Bing Chat.
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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Imports

# Standard library imports

# Third party imports
import numpy as np
from topollm.compute_embeddings.embedding_extractor.EmbeddingExtractorProtocol import (
    EmbeddingExtractor,
)
from transformers.configuration_utils import PretrainedConfig
import transformers.modeling_outputs

# Local imports

# Local imports
from topollm.compute_embeddings.embedding_extractor.LayerAggregator import (
    ConcatenateLayerAggregator,
    LayerAggregator,
    MeanLayerAggregator,
)
from topollm.compute_embeddings.embedding_extractor.LayerExtractor import (
    LayerExtractor,
    LayerExtractorFromIndices,
)
from topollm.config_classes.embeddings.EmbeddingExtractionConfig import (
    EmbeddingExtractionConfig,
)
from topollm.config_classes.enums import AggregationType

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class TokenLevelEmbeddingExtractor:
    """
    Implementation of the EmbeddingExtractor protocol
    which extracts token level embeddings.
    """

    def __init__(
        self,
        layer_extractor: LayerExtractor,
        layer_aggregator: LayerAggregator,
        embedding_dimension: int,
    ):
        self.layer_extractor = layer_extractor
        self.layer_aggregator = layer_aggregator
        self.embedding_dimension = embedding_dimension

    def extract_embeddings_from_model_outputs(
        self,
        model_outputs: transformers.modeling_outputs.BaseModelOutput,
    ) -> np.ndarray:
        # Ensure the model outputs hidden states
        if not hasattr(
            model_outputs,
            "hidden_states",
        ):
            raise ValueError("Model outputs do not contain 'hidden_states'")

        hidden_states = (
            model_outputs.hidden_states
        )  # Assuming this is a tuple of tensors

        if hidden_states is None:
            raise ValueError(
                f"'hidden_states' is None. "
                f"Did you call the model with 'output_hidden_states=True'?"
            )

        # Extract specified layers
        layers_to_extract = self.layer_extractor.extract_layers_from_model_outputs(
            hidden_states=hidden_states,
        )

        # Aggregate the extracted layers
        embeddings = self.layer_aggregator.aggregate_layers(
            layers_to_extract=layers_to_extract,
        )

        return embeddings.cpu().numpy()


def get_embedding_extractor(
    embedding_extraction_config: EmbeddingExtractionConfig,
    model_hidden_size: int,
) -> EmbeddingExtractor:
    layer_extractor = LayerExtractorFromIndices(
        layer_indices=embedding_extraction_config.layer_indices,
    )

    if embedding_extraction_config.aggregation == AggregationType.MEAN:
        layer_aggregator = MeanLayerAggregator()
        embedding_dimension = model_hidden_size
    elif embedding_extraction_config.aggregation == AggregationType.CONCATENATE:
        layer_aggregator = ConcatenateLayerAggregator()

        # Note that the following dimension computation assumes that the
        # hidden size of the model is the same for all layers.
        embedding_dimension = model_hidden_size * len(
            embedding_extraction_config.layer_indices,
        )
    else:
        raise ValueError(
            f"Unknown aggregation method: "
            f"{embedding_extraction_config.aggregation = }",
        )

    embedding_extractor = TokenLevelEmbeddingExtractor(
        layer_extractor=layer_extractor,
        layer_aggregator=layer_aggregator,
        embedding_dimension=embedding_dimension,
    )

    return embedding_extractor

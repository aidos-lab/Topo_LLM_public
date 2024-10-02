# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (ruppik@hhu.de)
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

from topollm.compute_embeddings.embedding_extractor.embedding_extractor_token_level import (
    EmbeddingExtractorTokenLevel,
)
from topollm.compute_embeddings.embedding_extractor.layer_aggregator.layer_aggregator_concatenate import (
    LayerAggregatorConcatenate,
)
from topollm.compute_embeddings.embedding_extractor.layer_aggregator.layer_aggregator_mean import (
    LayerAggregatorMean,
)
from topollm.compute_embeddings.embedding_extractor.layer_extractor.layer_extractor_from_indices import (
    LayerExtractorFromIndices,
)
from topollm.compute_embeddings.embedding_extractor.protocol import (
    EmbeddingExtractor,
)
from topollm.config_classes.embeddings.embedding_extraction_config import (
    EmbeddingExtractionConfig,
)
from topollm.typing.enums import AggregationType


def get_embedding_extractor(
    embedding_extraction_config: EmbeddingExtractionConfig,
    model_hidden_size: int,
) -> EmbeddingExtractor:
    """Create an embedding extractor."""
    layer_extractor = LayerExtractorFromIndices(
        layer_indices=embedding_extraction_config.layer_indices,
    )

    if embedding_extraction_config.aggregation == AggregationType.MEAN:
        layer_aggregator = LayerAggregatorMean()
        embedding_dimension = model_hidden_size
    elif embedding_extraction_config.aggregation == AggregationType.CONCATENATE:
        layer_aggregator = LayerAggregatorConcatenate()

        # Note that the following dimension computation assumes that the
        # hidden size of the model is the same for all layers.
        embedding_dimension = model_hidden_size * len(
            embedding_extraction_config.layer_indices,
        )
    else:
        msg = f"Unknown aggregation method: {embedding_extraction_config.aggregation = }"
        raise ValueError(
            msg,
        )

    embedding_extractor = EmbeddingExtractorTokenLevel(
        layer_extractor=layer_extractor,
        layer_aggregator=layer_aggregator,
        embedding_dimension=embedding_dimension,
    )

    return embedding_extractor

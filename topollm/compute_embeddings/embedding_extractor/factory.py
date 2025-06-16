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

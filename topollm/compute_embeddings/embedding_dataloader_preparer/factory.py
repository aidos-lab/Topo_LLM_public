"""Factory for creating embedding dataloader preparers."""

from topollm.compute_embeddings.embedding_dataloader_preparer.embedding_dataloader_preparer_context import (
    EmbeddingDataLoaderPreparerContext,
)
from topollm.compute_embeddings.embedding_dataloader_preparer.embedding_dataloader_preparer_huggingface import (
    EmbeddingDataLoaderPreparerHuggingfaceWithTokenization,
)
from topollm.compute_embeddings.embedding_dataloader_preparer.protocol import (
    EmbeddingDataLoaderPreparer,
)


def get_embedding_dataloader_preparer(
    preparer_context: EmbeddingDataLoaderPreparerContext,
) -> EmbeddingDataLoaderPreparer:
    """Instantiate dataloader preparers based on the dataset type.

    Returns
    -------
        An instance of a DatasetPreparer subclass.

    """
    result = EmbeddingDataLoaderPreparerHuggingfaceWithTokenization(
        preparer_context=preparer_context,
    )

    return result

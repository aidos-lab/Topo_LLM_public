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

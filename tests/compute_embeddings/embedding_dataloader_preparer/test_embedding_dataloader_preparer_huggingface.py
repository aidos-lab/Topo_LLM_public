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


import logging

import pytest

from topollm.compute_embeddings.embedding_dataloader_preparer.embedding_dataloader_preparer_huggingface import (
    EmbeddingDataLoaderPreparerHuggingfaceWithTokenization,
)


@pytest.mark.uses_transformers_models
def test_EmbeddingDataLoaderPreparerHuggingface(  # noqa: N802 - This is the name of a class
    embedding_dataloader_preparer_huggingface: EmbeddingDataLoaderPreparerHuggingfaceWithTokenization,
    logger_fixture: logging.Logger,
) -> None:
    """Test the EmbeddingDataLoaderPreparerHuggingface class."""
    dataloader = embedding_dataloader_preparer_huggingface.get_dataloader()

    assert dataloader is not None  # noqa: S101 - pytest assert

    # Test the length function
    length: int = len(embedding_dataloader_preparer_huggingface)
    logger_fixture.info(
        msg=f"{length = }",  # noqa: G004 - low overhead
    )

    assert length > 0  # noqa: S101 - pytest assert

"""Tests for the EmbeddingDataLoaderPreparerHuggingface class."""

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

"""Fixtures for the embedding dataloader preparer tests."""

from functools import partial

import pytest
import torch
import transformers

from topollm.compute_embeddings.collator.collate_batch_for_embedding import (
    collate_batch_and_move_to_device,
)
from topollm.compute_embeddings.embedding_dataloader_preparer.embedding_dataloader_preparer_context import (
    EmbeddingDataLoaderPreparerContext,
)
from topollm.compute_embeddings.embedding_dataloader_preparer.factory import get_embedding_dataloader_preparer
from topollm.compute_embeddings.embedding_dataloader_preparer.protocol import (
    EmbeddingDataLoaderPreparer,
)
from topollm.config_classes.data.data_config import DataConfig
from topollm.config_classes.embeddings.embeddings_config import EmbeddingsConfig
from topollm.config_classes.tokenizer.tokenizer_config import TokenizerConfig


@pytest.fixture(
    scope="session",
)
def preparer_context(
    device_fixture: torch.device,
    data_config: DataConfig,
    embeddings_config: EmbeddingsConfig,
    tokenizer_config: TokenizerConfig,
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
) -> EmbeddingDataLoaderPreparerContext:
    """Return a context for embedding dataloader preparers."""
    partial_collate_fn = partial(
        collate_batch_and_move_to_device,
        device=device_fixture,
        model_input_names=tokenizer.model_input_names,
    )

    result = EmbeddingDataLoaderPreparerContext(
        data_config=data_config,
        embeddings_config=embeddings_config,
        tokenizer_config=tokenizer_config,
        tokenizer=tokenizer,
        collate_fn=partial_collate_fn,
    )

    return result


@pytest.fixture(
    scope="session",
)
def embedding_dataloader_preparer_huggingface(
    preparer_context: EmbeddingDataLoaderPreparerContext,
) -> EmbeddingDataLoaderPreparer:
    """Return an instance of the EmbeddingDataLoaderPreparerHuggingface."""
    result: EmbeddingDataLoaderPreparer = get_embedding_dataloader_preparer(
        preparer_context=preparer_context,
    )

    return result

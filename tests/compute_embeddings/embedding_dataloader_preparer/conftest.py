"""Fixtures for the embedding dataloader preparer tests."""

import logging
from functools import partial
from typing import TYPE_CHECKING

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
from topollm.config_classes.language_model.language_model_config import LanguageModelConfig
from topollm.config_classes.tokenizer.tokenizer_config import TokenizerConfig
from topollm.model_handling.prepare_loaded_model_container import (
    prepare_device_and_tokenizer_and_model_from_language_model_config,
)
from topollm.typing.enums import PreferredTorchBackend, Verbosity

if TYPE_CHECKING:
    from collections.abc import Callable

    from topollm.model_handling.loaded_model_container import LoadedModelContainer


@pytest.fixture(
    scope="session",
)
def preparer_context(  # noqa: PLR0913 - we need all these parameters
    device_fixture: torch.device,
    data_config: DataConfig,
    language_model_config: LanguageModelConfig,
    embeddings_config: EmbeddingsConfig,
    tokenizer_config: TokenizerConfig,
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
    verbosity: Verbosity,
    logger_fixture: logging.Logger,
) -> EmbeddingDataLoaderPreparerContext:
    """Return a context for embedding dataloader preparers."""
    loaded_model_container: LoadedModelContainer = prepare_device_and_tokenizer_and_model_from_language_model_config(
        language_model_config=language_model_config,
        tokenizer_config=tokenizer_config,
        preferred_torch_backend=PreferredTorchBackend.CPU,
        verbosity=verbosity,
        logger=logger_fixture,
    )

    partial_collate_fn: Callable[[list], dict] = partial(
        collate_batch_and_move_to_device,
        device=device_fixture,
        loaded_model_container=loaded_model_container,
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

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

"""Fixtures for the embedding dataloader preparer tests."""

from functools import partial

import pytest
import torch
import transformers

from topollm.compute_embeddings.collate_batch_for_embedding import (
    collate_batch_and_move_to_device,
)
from topollm.compute_embeddings.embedding_dataloader_preparer.embedding_dataloader_preparer_context import (
    EmbeddingDataLoaderPreparerContext,
)
from topollm.compute_embeddings.embedding_dataloader_preparer.embedding_dataloader_preparer_huggingface import (
    EmbeddingDataLoaderPreparerHuggingface,
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

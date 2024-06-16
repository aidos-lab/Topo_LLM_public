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

import logging

import pytest

from topollm.compute_embeddings.embedding_dataloader_preparer.embedding_dataloader_preparer_huggingface import (
    EmbeddingDataLoaderPreparerHuggingface,
)


@pytest.mark.uses_transformers_models()
def test_EmbeddingDataLoaderPreparerHuggingface(
    embedding_dataloader_preparer_huggingface: EmbeddingDataLoaderPreparerHuggingface,
    logger_fixture: logging.Logger,
) -> None:
    dataloader = embedding_dataloader_preparer_huggingface.prepare_dataloader()

    assert dataloader is not None  # noqa: S101 - pytest assert

    # Test the length function
    length = len(embedding_dataloader_preparer_huggingface)
    logger_fixture.info(
        f"{length = }",  # noqa: G004 - low overhead
    )

    assert length > 0  # noqa: S101 - pytest assert

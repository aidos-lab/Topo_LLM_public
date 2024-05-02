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

"""Factory for creating embedding dataloader preparers."""

from topollm.compute_embeddings.embedding_dataloader_preparer.embedding_dataloader_preparer_context import (
    EmbeddingDataLoaderPreparerContext,
)
from topollm.compute_embeddings.embedding_dataloader_preparer.embedding_dataloader_preparer_huggingface import (
    EmbeddingDataLoaderPreparerHuggingface,
)
from topollm.compute_embeddings.embedding_dataloader_preparer.protocol import (
    EmbeddingDataLoaderPreparer,
)
from topollm.typing.enums import DatasetType


def get_embedding_dataloader_preparer(
    preparer_context: EmbeddingDataLoaderPreparerContext,
) -> EmbeddingDataLoaderPreparer:
    """Instantiate dataloader preparers based on the dataset type.

    Returns
    -------
        An instance of a DatasetPreparer subclass.

    """
    if preparer_context.data_config.dataset_type == DatasetType.HUGGINGFACE_DATASET:
        result = EmbeddingDataLoaderPreparerHuggingface(
            preparer_context=preparer_context,
        )
    else:
        msg = f"Unsupported {preparer_context.data_config.dataset_type = }"
        raise ValueError(msg)

    return result

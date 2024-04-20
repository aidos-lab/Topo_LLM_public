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

from topollm.compute_embeddings.embedding_dataloader_preparer.EmbeddingDataLoaderPreparerABC import (
    EmbeddingDataLoaderPreparer,
)
from topollm.compute_embeddings.embedding_dataloader_preparer.EmbeddingDataLoaderPreparerContext import (
    EmbeddingDataLoaderPreparerContext,
)
from topollm.compute_embeddings.embedding_dataloader_preparer.EmbeddingDataLoaderPreparerHuggingface import (
    EmbeddingDataLoaderPreparerHuggingface,
)
from topollm.config_classes.enums import DatasetType


def get_embedding_dataloader_preparer(
    dataset_type: DatasetType,
    preparer_context: EmbeddingDataLoaderPreparerContext,
) -> EmbeddingDataLoaderPreparer:
    """Instantiate dataloader preparers based on the dataset type.

    Args:
    ----
        dataset_type:
            The type of dataset to prepare.
        config:
            Configuration object containing dataset and model settings.
        tokenizer:
            Tokenizer object for datasets that require tokenization.

    Returns:
    -------
        An instance of a DatasetPreparer subclass.

    """
    if dataset_type == DatasetType.HUGGINGFACE_DATASET:
        return EmbeddingDataLoaderPreparerHuggingface(
            preparer_context=preparer_context,
        )
    else:
        msg = f"Unsupported {dataset_type = }"
        raise ValueError(msg)

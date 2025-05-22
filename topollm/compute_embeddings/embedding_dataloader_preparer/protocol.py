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

"""Protocol for embedding dataset preparers."""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import datasets
import torch
import torch.utils.data

from topollm.compute_embeddings.embedding_dataloader_preparer.convert_dataset_entry_to_features_functions import (
    get_convert_dataset_entry_to_features_function,
)
from topollm.compute_embeddings.embedding_dataloader_preparer.embedding_dataloader_preparer_context import (
    EmbeddingDataLoaderPreparerContext,
)
from topollm.data_handling.dataset_preparer.factory import get_dataset_preparer

if TYPE_CHECKING:
    from collections.abc import Callable

    from topollm.data_handling.dataset_preparer.protocol import DatasetPreparer


class EmbeddingDataLoaderPreparer(ABC):
    """Abstract base class for embedding dataset preparers."""

    def __init__(
        self,
        preparer_context: EmbeddingDataLoaderPreparerContext,
    ) -> None:
        """Initialize the embedding dataset preparer."""
        self.preparer_context: EmbeddingDataLoaderPreparerContext = preparer_context

        self.dataset_preparer: DatasetPreparer = get_dataset_preparer(
            data_config=self.preparer_context.data_config,
            verbosity=self.preparer_context.verbosity,
            logger=self.preparer_context.logger,
        )

        self.convert_dataset_entry_to_features_function: Callable = get_convert_dataset_entry_to_features_function(
            data_config=self.preparer_context.data_config,
        )

        # These private attributes will hold the dataset and dataloader
        self._dataset: datasets.Dataset | None = None
        self._dataset_tokenized: datasets.Dataset | None = None
        self._dataloader: torch.utils.data.DataLoader | None = None

    @property
    def logger(
        self,
    ) -> logging.Logger:
        return self.preparer_context.logger

    @property
    def verbosity(
        self,
    ) -> int:
        return self.preparer_context.verbosity

    @abstractmethod
    def get_dataset(
        self,
    ) -> datasets.Dataset:
        """Prepare a dataset."""
        ...  # pragma: no cover

    @abstractmethod
    def get_dataset_tokenized(
        self,
    ) -> datasets.Dataset:
        """Tokenize the prepared dataset."""
        ...  # pragma: no cover

    @abstractmethod
    def get_dataloader(
        self,
    ) -> torch.utils.data.DataLoader:
        """Get the dataloader based on the tokenized dataset."""
        ...  # pragma: no cover

    @property
    @abstractmethod
    def sequence_length(
        self,
    ) -> int:
        """Return the sequence length of the dataset."""
        ...  # pragma: no cover

    @abstractmethod
    def __len__(
        self,
    ) -> int:
        """Return the number of samples in the dataset."""
        ...  # pragma: no cover

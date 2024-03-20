# coding=utf-8
#
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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Imports

# Standard library imports
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

# Third party imports
import torch
import torch.utils.data
from transformers import BatchEncoding, PreTrainedTokenizer, PreTrainedTokenizerFast

# Local imports
from topollm.data_handling.HuggingfaceDatasetPreparer import HuggingfaceDatasetPreparer
from topollm.config_classes.EmbeddingsConfig import EmbeddingsConfig
from topollm.config_classes.DataConfig import DataConfig

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@dataclass
class EmbeddingDataLoaderPreparerContext:
    """Encapsulates the context needed for preparing dataloaders."""

    data_config: DataConfig
    embeddings_config: EmbeddingsConfig
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
    collate_fn: Callable[[list], dict]
    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger(__name__),
    )
    verbosity: int = 1


class EmbeddingDataLoaderPreparer(ABC):
    """Abstract base class for embedding dataset preparers."""

    def __init__(
        self,
        preparer_context: EmbeddingDataLoaderPreparerContext,
    ):
        self.preparer_context = preparer_context

        self.dataset_preparer = HuggingfaceDatasetPreparer(
            data_config=self.preparer_context.data_config,
            verbosity=self.preparer_context.verbosity,
            logger=self.preparer_context.logger,
        )

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
    def prepare_dataloader(
        self,
    ) -> torch.utils.data.DataLoader:
        """Loads a dataset and prepares a dataloader."""
        pass

    @staticmethod
    def convert_dataset_entry_to_features(
        dataset_entry: dict,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        column_name: str = "text",
        max_length: int = 512,
    ) -> BatchEncoding:
        """
        Convert dataset entires/examples to features
        by tokenizing the text and padding/truncating to a maximum length.
        """

        features = tokenizer(
            dataset_entry[column_name],
            max_length=max_length,
            padding="max_length",
            truncation="longest_first",
        )

        return features

    @property
    @abstractmethod
    def sequence_length(self) -> int:
        """Returns the sequence length of the dataset."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        pass

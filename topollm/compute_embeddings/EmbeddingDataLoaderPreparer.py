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
from dataclasses import dataclass, field
from functools import partial
import logging
from abc import ABC, abstractmethod
from multiprocessing.spawn import prepare
from typing import Callable

# Third party imports
import datasets
from sympy import sequence
import torch
import torch.utils.data
from transformers import BatchEncoding, PreTrainedTokenizer, PreTrainedTokenizerFast

# Local imports
from topollm.config_classes.Configs import DataConfig, EmbeddingsConfig
from topollm.config_classes.enums import DatasetType

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


class HuggingfaceEmbeddingDataLoaderPreparer(EmbeddingDataLoaderPreparer):
    def load_dataset_dict(
        self,
    ) -> datasets.DatasetDict:
        """Loads the dataset based from huggingface datasets based on configuration."""
        dataset_dict = datasets.load_dataset(
            self.preparer_context.data_config.dataset_identifier,
            trust_remote_code=True,
        )

        if not isinstance(
            dataset_dict,
            datasets.DatasetDict,
        ):
            raise ValueError(
                f"Expected {dataset_dict = } " f"to be a {datasets.DatasetDict = }"
            )

        return dataset_dict

    @property
    def sequence_length(
        self,
    ) -> int:
        return self.preparer_context.embeddings_config.max_length

    def __len__(
        self,
    ) -> int:
        """Returns the number of samples in the dataset."""
        if not hasattr(
            self,
            "dataset_length",
        ):
            raise ValueError(
                "The dataset length is not available. "
                "Please call prepare_dataloader() first."
            )

        return self.dataset_length

    def select_dataset(
        self,
        dataset_dict: datasets.DatasetDict,
    ) -> datasets.Dataset:
        # Select the dataset split to use
        dataset: datasets.Dataset = dataset_dict[
            self.preparer_context.data_config.split
        ]

        # Truncate the dataset to the specified number of samples
        dataset = dataset.select(
            indices=range(self.preparer_context.data_config.number_of_samples),
        )

        self.dataset_length = len(dataset)

        return dataset

    def create_dataset_tokenized(
        self,
        dataset: datasets.Dataset,
    ) -> datasets.Dataset:
        """Tokenizes dataset."""
        # Make a partial function for mapping tokenizer over the dataset
        partial_map_fn = partial(
            self.convert_dataset_entry_to_features,
            tokenizer=self.preparer_context.tokenizer,
            column_name=self.preparer_context.data_config.column_name,
            max_length=self.sequence_length,
        )

        dataset_tokenized = dataset.map(
            partial_map_fn,
            batched=True,
            batch_size=self.preparer_context.embeddings_config.dataset_map.batch_size,
            num_proc=self.preparer_context.embeddings_config.dataset_map.num_proc,
        )

        return dataset_tokenized

    def create_dataloader_from_tokenized_dataset(
        self,
        dataset_tokenized: datasets.Dataset,
    ) -> torch.utils.data.DataLoader:
        # The mapped dataset has the input_ids and attention_mask
        # as lists of integers, but we want to convert them to torch tensors
        # to use them as model input.
        # We will take care of this in the collate function of the DataLoader,
        # which will also move the data to the appropriate device.
        #
        # An alternative way to set the format of the dataset to torch tensors
        # is given below:
        #
        # dataset_tokenized.set_format(
        #     type="torch",
        #     columns=[
        #         "input_ids",
        #         "attention_mask",
        #     ],
        # )

        dataloader = torch.utils.data.DataLoader(
            dataset_tokenized,  # type: ignore
            batch_size=self.preparer_context.embeddings_config.batch_size,
            shuffle=False,
            collate_fn=self.preparer_context.collate_fn,
            num_workers=self.preparer_context.embeddings_config.num_workers,
        )

        return dataloader

    def prepare_dataloader(
        self,
    ) -> torch.utils.data.DataLoader:
        """Loads and prepares a dataset."""
        dataset_dict = self.load_dataset_dict()
        dataset = self.select_dataset(
            dataset_dict=dataset_dict,
        )
        dataset_tokenized = self.create_dataset_tokenized(
            dataset=dataset,
        )
        dataloader = self.create_dataloader_from_tokenized_dataset(
            dataset_tokenized=dataset_tokenized,
        )

        return dataloader


def get_embedding_dataloader_preparer(
    dataset_type: DatasetType,
    preparer_context: EmbeddingDataLoaderPreparerContext,
) -> EmbeddingDataLoaderPreparer:
    """Factory function to instantiate dataset preparers based on the dataset type.

    Args:
        dataset_type:
            The type of dataset to prepare.
        config:
            Configuration object containing dataset and model settings.
        tokenizer:
            Tokenizer object for datasets that require tokenization.

    Returns:
        An instance of a DatasetPreparer subclass.
    """
    if dataset_type == DatasetType.HUGGINGFACE_DATASET:
        return HuggingfaceEmbeddingDataLoaderPreparer(
            preparer_context=preparer_context,
        )
    # Extendable to other dataset types
    # elif dataset_type == "unified_format":
    #     return ImageDatasetPreparer(config)
    else:
        raise ValueError(f"Unsupported {dataset_type = }")

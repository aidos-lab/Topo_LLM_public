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
import nltk
from abc import ABC, abstractmethod

import torch
import torch.utils.data
from transformers import BatchEncoding, PreTrainedTokenizer, PreTrainedTokenizerFast

from topollm.compute_embeddings.embedding_dataloader_preparer.embedding_dataloader_preparer_context import (
    EmbeddingDataLoaderPreparerContext,
)
from topollm.data_handling.dataset_preparer.factory import get_dataset_preparer
from topollm.data_handling.dataset_preparer.protocol import DatasetPreparer


class EmbeddingDataLoaderPreparer(ABC):
    """Abstract base class for embedding dataset preparers."""

    def __init__(
        self,
        preparer_context: EmbeddingDataLoaderPreparerContext,
    ) -> None:
        self.preparer_context = preparer_context

        self.dataset_preparer: DatasetPreparer = get_dataset_preparer(
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

    @staticmethod
    def convert_dataset_entry_to_features(
        dataset_entry: dict,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        column_name: str = "text",
        max_length: int = 512,
    ) -> BatchEncoding:
        """Convert dataset entires/examples to features by tokenizing the text and padding/truncating to a maximum length."""
        features = tokenizer(
            dataset_entry[column_name],
            max_length=max_length,
            padding="max_length",
            truncation="longest_first",
        )

        return features

    @staticmethod
    def convert_dataset_entry_to_features_named_entity(
            dataset_entry: dict,
            tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
            column_name: str = "text",
            max_length: int = 512,
    ) -> BatchEncoding:
        """Convert dataset entires/examples to features by tokenizing the text and padding/truncating to a maximum length."""
        split_words = [nltk.word_tokenize(sent) for sent in dataset_entry[column_name]]
        features = tokenizer(
            split_words,
            max_length=max_length,
            padding="max_length",
            truncation="longest_first",
            is_split_into_words=True,
            add_prefix_space=True
        )
        word_ids = [features.word_ids(batch_index=i) for i in range(len(features))]

        dataset_tokenized = features.input_ids

        pos_tag = [nltk.pos_tag(sent) for sent in split_words]


        all_word_tags_one_sentence_tokens = []

        for sentence_idx in range(len(dataset_tokenized)):
            word_tags_one_sentence = pos_tag[sentence_idx]
            word_tags_one_sentence = [word_tags_one_sentence[i][1] for i in range(len(word_tags_one_sentence))]
            word_ids_one_sentence = word_ids[sentence_idx]

            print(
                f"------------------------{(word_ids_one_sentence)}-----------------------:")
            print(
                f"------------------------{(word_tags_one_sentence)}-----------------------:")

            word_tags_one_sentence_tokens = []
            for i in word_ids_one_sentence:
                if i != None:
                    word_tags_one_sentence_tokens.append(word_tags_one_sentence[i])
                else:
                    word_tags_one_sentence_tokens.append(None)
            all_word_tags_one_sentence_tokens.append(word_tags_one_sentence_tokens)

        features['POS'] = all_word_tags_one_sentence_tokens

        return features

    @abstractmethod
    def prepare_dataloader(
        self,
    ) -> torch.utils.data.DataLoader:
        """Load a dataset and prepare a dataloader."""
        pass  # pragma: no cover

    @property
    @abstractmethod
    def sequence_length(
        self,
    ) -> int:
        """Return the sequence length of the dataset."""
        # pragma: no cover

    @abstractmethod
    def __len__(
        self,
    ) -> int:
        """Return the number of samples in the dataset."""
        # pragma: no cover

# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Matthias Ruppik (mail@ruppik.net)
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

"""Functions for converting dataset entries to features."""

from collections.abc import Callable

import nltk
from transformers import BatchEncoding, PreTrainedTokenizer, PreTrainedTokenizerFast

from topollm.config_classes.data.data_config import DataConfig
from topollm.typing.enums import DatasetType


def get_convert_dataset_entry_to_features_function(
    data_config: DataConfig,
) -> Callable[
    ...,
    BatchEncoding,
]:
    """Get the function to convert a dataset entry to features."""
    match data_config.dataset_type:
        case DatasetType.HUGGINGFACE_DATASET:
            dataset_entry_to_features_function = convert_dataset_entry_to_features
        case DatasetType.HUGGINGFACE_DATASET_NAMED_ENTITY:
            dataset_entry_to_features_function = convert_dataset_entry_to_features_named_entity
        case DatasetType.SETSUMBT_DATALOADERS_PROCESSED | DatasetType.TRIPPY_DATALOADERS_PROCESSED:
            # In this mode, the dataset entries are already tokenized and can be directly used as features.
            dataset_entry_to_features_function = convert_dataset_entry_to_features_do_nothing
        case _:
            msg: str = f"Unsupported {data_config.dataset_type = }"
            raise ValueError(msg)

    return dataset_entry_to_features_function


def convert_dataset_entry_to_features_do_nothing(
    dataset_entry: dict,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    column_name: str = "text",
    max_length: int = 512,
) -> BatchEncoding:
    """Do nothing."""
    return BatchEncoding(
        data=dataset_entry,
    )


def convert_dataset_entry_to_features(
    dataset_entry: dict,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    column_name: str = "text",
    max_length: int = 512,
) -> BatchEncoding:
    """Convert dataset entries/examples to features by tokenizing the text and padding/truncating to a maximum length."""
    features: BatchEncoding = tokenizer(
        dataset_entry[column_name],
        max_length=max_length,
        padding="max_length",
        truncation="longest_first",
    )

    return features


def convert_dataset_entry_to_features_named_entity(
    dataset_entry: dict,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    column_name: str = "text",
    max_length: int = 512,
) -> BatchEncoding:
    """Convert dataset entries/examples to features by tokenizing the text and padding/truncating to a maximum length."""
    split_words: list[list[str]] = [
        nltk.word_tokenize(
            text=sent,
        )
        for sent in dataset_entry[column_name]
    ]

    features: BatchEncoding = tokenizer(
        split_words,
        max_length=max_length,
        padding="max_length",
        truncation="longest_first",
        is_split_into_words=True,
    )

    word_ids = [features.word_ids(batch_index=i) for i in range(len(split_words))]

    dataset_tokenized = features.input_ids

    pos_tag = [nltk.pos_tag(tokens=sent) for sent in split_words]

    all_word_tags_one_sentence_tokens = []

    for sentence_idx in range(len(dataset_tokenized)):
        word_tags_one_sentence = pos_tag[sentence_idx]
        word_tags_one_sentence = [word_tags_one_sentence[i][1] for i in range(len(word_tags_one_sentence))]
        word_ids_one_sentence = word_ids[sentence_idx]

        word_tags_one_sentence_tokens = []
        for i in word_ids_one_sentence:
            if i is not None:
                word_tags_one_sentence_tokens.append(word_tags_one_sentence[i])
            else:
                word_tags_one_sentence_tokens.append(None)
        all_word_tags_one_sentence_tokens.append(word_tags_one_sentence_tokens)

    features["POS"] = all_word_tags_one_sentence_tokens

    return features

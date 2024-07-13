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

"""Prepare the model input for finetuning."""

from collections.abc import Callable
from functools import partial

import datasets
import transformers

from topollm.config_classes.finetuning.finetuning_config import FinetuningConfig
from topollm.typing.enums import TaskType


def prepare_model_input(
    train_dataset: datasets.Dataset,
    eval_dataset: datasets.Dataset,
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
    finetuning_config: FinetuningConfig,
) -> tuple[
    datasets.Dataset,
    datasets.Dataset,
]:
    """Prepare the model input for finetuning.

    NOTE: This implementation assumes that train and eval datasets use the same column name.
    """
    tokenize_function = get_tokenize_function(
        tokenizer=tokenizer,  # type: ignore - ignoring the requirement for PreTrainedTokenizerFast here
        finetuning_config=finetuning_config,
    )

    train_dataset_mapped = train_dataset.map(
        tokenize_function,
        batched=True,
    )
    eval_dataset_mapped = eval_dataset.map(
        tokenize_function,
        batched=True,
    )

    return train_dataset_mapped, eval_dataset_mapped


def get_tokenize_function(
    tokenizer: transformers.PreTrainedTokenizerFast,
    finetuning_config: FinetuningConfig,
) -> Callable[
    [
        dict[
            str,
            list,
        ],
    ],
    transformers.tokenization_utils_base.BatchEncoding,
]:
    """Get the tokenize function based on the finetuning config."""
    if finetuning_config.base_model.task_type in (TaskType.CAUSAL_LM, TaskType.MASKED_LM):
        tokenize_function = partial(
            standard_tokenize_function,
            tokenizer=tokenizer,
            finetuning_config=finetuning_config,
        )
    elif finetuning_config.base_model.task_type == TaskType.TOKEN_CLASSIFICATION:
        tokenize_function = partial(
            tokenize_split_into_words_input_and_align_labels,
            tokenizer=tokenizer,
            finetuning_config=finetuning_config,
        )
    else:
        msg = f"Task type {finetuning_config.base_model.task_type} is not supported."
        raise ValueError(
            msg,
        )

    return tokenize_function


def standard_tokenize_function(
    dataset_entries: dict[
        str,
        list,
    ],
    tokenizer: transformers.PreTrainedTokenizerFast,
    finetuning_config: FinetuningConfig,
) -> transformers.tokenization_utils_base.BatchEncoding:
    """Tokenize the dataset entries."""
    column_name: str = finetuning_config.finetuning_datasets.train_dataset.column_name

    result = tokenizer(
        dataset_entries[column_name],
        truncation=True,
        padding="max_length",
        max_length=finetuning_config.tokenizer.max_length,
        return_special_tokens_mask=finetuning_config.tokenizer.return_special_tokens_mask,
    )

    return result


def tokenize_split_into_words_input_and_align_labels(
    dataset_entries: dict[
        str,
        list,
    ],
    tokenizer: transformers.PreTrainedTokenizerFast,
    finetuning_config: FinetuningConfig,
) -> transformers.tokenization_utils_base.BatchEncoding:
    """Tokenize the dataset entries and align the labels.

    This is used if the input actually has not been tokenized yet
    and you will need to set `is_split_into_words=True` to tokenize the words into subwords.
    A single word corresponding to a single label may now be split into two subwords.
    We need to realign the tokens and labels by:
    - Mapping all tokens to their corresponding word with the word_ids method.
    - Assigning the label -100 to the special tokens [CLS] and [SEP]
      so they are ignored by the PyTorch loss function (see CrossEntropyLoss).
    - Only labeling the first token of a given word.
      Assign -100 to other subtokens from the same word.

    Code inspired by:
    https://huggingface.co/docs/transformers/en/tasks/token_classification
    """
    column_name: str = finetuning_config.finetuning_datasets.train_dataset.column_name

    tokenized_inputs = tokenizer(
        dataset_entries[column_name],
        truncation=True,
        padding="max_length",
        max_length=finetuning_config.tokenizer.max_length,
        is_split_into_words=True,
    )

    feature_column_name: str = finetuning_config.finetuning_datasets.train_dataset.feature_column_name

    labels = []
    for i, label in enumerate(
        dataset_entries[feature_column_name],
    ):
        word_ids = tokenized_inputs.word_ids(
            batch_index=i,
        )  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:  # Set the special tokens to -100.
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels

    return tokenized_inputs

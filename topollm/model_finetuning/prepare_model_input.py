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

import datasets
import transformers

from topollm.config_classes.finetuning.finetuning_config import FinetuningConfig


def prepare_model_input(
    train_dataset: datasets.Dataset,
    eval_dataset: datasets.Dataset,
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
    finetuning_config: FinetuningConfig,
) -> tuple[
    datasets.Dataset,
    datasets.Dataset,
]:
    """Prepare the model input for finetuning."""

    # TODO(Ben): Update this to work with text split into tokens.
    # Exception has occurred: TypeError
    # TextEncodeInput must be Union[TextInputSequence, Tuple[InputSequence, InputSequence]]

    def tokenize_function(
        dataset_entries: dict[
            str,
            list,
        ],
    ) -> transformers.tokenization_utils_base.BatchEncoding:
        """Tokenize the dataset entries.

        NOTE: This implementation assumes that train and eval datasets use the same column name.
        """
        column_name = finetuning_config.finetuning_datasets.train_dataset.column_name

        result = tokenizer(
            dataset_entries[column_name],
            padding="max_length",
            truncation=True,
            max_length=finetuning_config.max_length,
        )

        return result

    train_dataset_mapped = train_dataset.map(
        tokenize_function,
        batched=True,
    )
    eval_dataset_mapped = eval_dataset.map(
        tokenize_function,
        batched=True,
    )

    return train_dataset_mapped, eval_dataset_mapped

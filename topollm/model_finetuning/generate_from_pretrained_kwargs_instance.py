# Copyright 2024
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

import logging

import datasets

from topollm.config_classes.finetuning.finetuning_config import FinetuningConfig
from topollm.model_handling.model.token_classification_from_pretrained_kwargs import (
    TokenClassificationFromPretrainedKwargs,
)
from topollm.typing.enums import TaskType, Verbosity

default_logger = logging.getLogger(__name__)


def generate_from_pretrained_kwargs_instance(
    finetuning_config: FinetuningConfig,
    label_list: list[str] | None,
) -> TokenClassificationFromPretrainedKwargs | None:
    """Generate the from_pretrained_kwargs_instance for token classification."""
    match finetuning_config.base_model.task_type:
        case TaskType.CAUSAL_LM | TaskType.MASKED_LM:
            from_pretrained_kwargs_instance = None
        case TaskType.TOKEN_CLASSIFICATION:
            if label_list is None:
                msg = (
                    "Could not generate from_pretrained_kwargs_instance for token classification, "
                    "since the `label_list` is None."
                )
                raise ValueError(msg)

            from_pretrained_kwargs_instance = TokenClassificationFromPretrainedKwargs(
                num_labels=len(label_list),
                id2label=dict(enumerate(label_list)),
                label2id={label: i for i, label in enumerate(label_list)},
            )
        case _:
            msg = f"Unknown {finetuning_config.base_model.task_type = }"
            raise ValueError(msg)
    return from_pretrained_kwargs_instance


def extract_label_list(
    finetuning_config: FinetuningConfig,
    train_dataset: datasets.Dataset,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> list[str] | None:
    """Extract the label list from the train_dataset."""
    feature_column_name: str = finetuning_config.finetuning_datasets.train_dataset.feature_column_name

    try:
        label_list = train_dataset.features[feature_column_name].feature.names
    except AttributeError:
        if verbosity >= Verbosity.NORMAL:
            logger.warning(
                "Could not extract label list from train_dataset.features[%s].feature.names.",
                feature_column_name,
            )
        label_list = None
    except KeyError:
        if verbosity >= Verbosity.NORMAL:
            logger.warning(
                "Could not find column train_dataset.features[%s]",
                feature_column_name,
            )
        label_list = None

    return label_list

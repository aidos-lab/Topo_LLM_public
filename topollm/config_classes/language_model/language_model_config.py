# Copyright 2024-2025
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

"""Configuration for specifying the language model."""

import pathlib

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel
from topollm.config_classes.constants import ITEM_SEP, KV_SEP, NAME_PREFIXES
from topollm.config_classes.language_model.tokenizer_modifier.tokenizer_modifier_config import (
    TokenizerModifierConfig,
)
from topollm.typing.enums import DescriptionType, DropoutMode, LMmode, TaskType


class DropoutProbabilities(ConfigBaseModel):
    """Configuration for specifying dropout probabilities."""

    hidden_dropout_prob: float = Field(
        default=0.1,
        title="Hidden dropout probability.",
        description="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
    )

    attention_probs_dropout_prob: float = Field(
        default=0.1,
        title="Attention dropout probability.",
        description="The dropout ratio for the attention probabilities.",
    )

    classifier_dropout: float | None = Field(
        default=None,
        title="Classifier dropout probability.",
        description="The dropout ratio for the classification head.",
    )

    def get_config_description(
        self,
        description_type: DescriptionType = DescriptionType.LONG,
        short_description_separator: str = "-",
    ) -> str:
        """Return a description of the configuration."""
        match description_type:
            case DescriptionType.LONG:
                description: str = (
                    f"{NAME_PREFIXES['hidden_dropout_prob']}"
                    f"{KV_SEP}"
                    f"{self.hidden_dropout_prob}"
                    f"{ITEM_SEP}"
                    f"{NAME_PREFIXES['attention_probs_dropout_prob']}"
                    f"{KV_SEP}"
                    f"{self.attention_probs_dropout_prob}"
                    f"{ITEM_SEP}"
                    f"{NAME_PREFIXES['classifier_dropout']}"
                    f"{KV_SEP}"
                    f"{self.classifier_dropout}"
                )
            case DescriptionType.SHORT:
                description: str = (
                    str(object=self.hidden_dropout_prob)
                    + short_description_separator
                    + str(object=self.attention_probs_dropout_prob)
                    + short_description_separator
                    + str(object=self.classifier_dropout)
                )
            case _:
                msg: str = f"Unknown {description_type = }"
                raise ValueError(
                    msg,
                )

        return description


class DropoutConfig(ConfigBaseModel):
    """Configuration for specifying dropout."""

    mode: DropoutMode = Field(
        default=DropoutMode.DEFAULTS,
        title="Dropout mode.",
        description="The dropout mode: Selecting 'DEFAULTS' will use the default parameters for the given model class.",
    )

    probabilities: DropoutProbabilities = Field(
        default_factory=DropoutProbabilities,
        title="Dropout probabilities.",
        description="The dropout probabilities.",
    )

    def get_config_description(
        self,
        description_type: DescriptionType = DescriptionType.LONG,
        short_description_separator: str = "-",
    ) -> str:
        match self.mode:
            case DropoutMode.DEFAULTS:
                match description_type:
                    case DescriptionType.LONG:
                        description: str = f"{NAME_PREFIXES['dropout_mode']}{KV_SEP}{str(object=self.mode)}"
                    case DescriptionType.SHORT:
                        description: str = str(object=self.mode)
            case DropoutMode.MODIFY_ROBERTA_DROPOUT_PARAMETERS:
                match description_type:
                    case DescriptionType.LONG:
                        description: str = (
                            f"{NAME_PREFIXES['dropout_mode']}{KV_SEP}{str(object=self.mode)}"
                            + ITEM_SEP
                            + self.probabilities.get_config_description(
                                description_type=description_type,
                                short_description_separator=short_description_separator,
                            )
                        )
                    case DescriptionType.SHORT:
                        description: str = (
                            # We do not use the dropout mode in the short description
                            self.probabilities.get_config_description(
                                description_type=description_type,
                                short_description_separator=short_description_separator,
                            )
                        )
                    case _:
                        msg: str = f"Unknown {description_type = }"
                        raise ValueError(
                            msg,
                        )
            case _:
                msg: str = f"Unknown {self.mode = }"
                raise ValueError(
                    msg,
                )

        return description


class LanguageModelConfig(ConfigBaseModel):
    """Configuration for specifying the language model."""

    checkpoint_no: int = Field(
        default=-1,
        title="Checkpoint number.",
        description="The checkpoint number.",
    )

    lm_mode: LMmode = Field(
        default=LMmode.MLM,
        title="Language model mode.",
        description="The language model mode.",
    )

    task_type: TaskType = Field(
        default=TaskType.MASKED_LM,
        title="Task type.",
        description="The task type.",
    )

    pretrained_model_name_or_path: str | pathlib.Path = Field(
        default="roberta-base",
        title="Model identifier for huggingface transformers model.",
        description="The model identifier for the huggingface transformers model to use for computing embeddings.",
    )

    model_log_file_path: str | pathlib.Path | None = Field(
        default=None,
        title="Optional path to a log file about the model training.",
        description="Path to a log file about the model training.",
    )

    manual_tokenizer_override_pretrained_model_name_or_path: str | pathlib.Path | None = Field(
        default=None,
        title="If this is not None, it will be used for loading the tokenizer.",
        description="Tokenizer which will be loaded instead of the one set in 'pretrained_model_name_or_path'.",
    )

    seed: int | None = Field(
        default=None,
        title="Random seed.",
        description="Random seed describing the model (for example, if it results from fine-tuning).",
    )

    short_model_name: str = Field(
        default="roberta-base",
        title="Short model name.",
        description="The short model name.",
    )

    # Note: There is a parameter in the DropoutConfig class to deactivate the dropout.
    dropout: DropoutConfig = Field(
        default_factory=DropoutConfig,
        title="Dropout configuration.",
        description="The configuration for specifying the dropout rate.",
    )

    tokenizer_modifier: TokenizerModifierConfig = Field(
        default_factory=TokenizerModifierConfig,
        description="The configuration for modifying the tokenizer.",
    )

    def get_config_description(
        self,
        description_type: DescriptionType = DescriptionType.LONG,
        short_description_separator: str = "-",
    ) -> str:
        """Construct and return the model description."""
        dropout_description: str = self.dropout.get_config_description(
            description_type=description_type,
            short_description_separator=short_description_separator,
        )

        match description_type:
            case DescriptionType.LONG:
                model_and_task_description_long: str = (
                    f"{NAME_PREFIXES['model']}"
                    f"{KV_SEP}"
                    f"{self.short_model_name}"
                    f"{ITEM_SEP}"
                    f"{NAME_PREFIXES['task_type']}"
                    f"{KV_SEP}"
                    f"{self.task_type}"
                )

                description: str = model_and_task_description_long + ITEM_SEP + dropout_description
            case DescriptionType.SHORT:
                model_and_task_description_short: str = (
                    f"{self.short_model_name}" + short_description_separator + f"{self.task_type}"
                )
                description: str = model_and_task_description_short + short_description_separator + dropout_description

        return description

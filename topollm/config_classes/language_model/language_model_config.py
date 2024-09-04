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

"""Configuration for specifying the language model."""

import pathlib

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel
from topollm.config_classes.constants import ITEM_SEP, KV_SEP, NAME_PREFIXES
from topollm.config_classes.language_model.tokenizer_modifier.tokenizer_modifier_config import (
    TokenizerModifierConfig,
)
from topollm.typing.enums import LMmode, TaskType


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

    manual_tokenizer_override_pretrained_model_name_or_path: str | pathlib.Path | None = Field(
        default=None,
        title="If this is not None, it will be used for loading the tokenizer.",
        description="Tokenizer which will be loaded instead of the one set in 'pretrained_model_name_or_path'.",
    )

    short_model_name: str = Field(
        default="roberta-base",
        title="Short model name.",
        description="The short model name.",
    )

    tokenizer_modifier: TokenizerModifierConfig = Field(
        default_factory=TokenizerModifierConfig,
        description="The configuration for modifying the tokenizer.",
    )

    @property
    def config_description(
        self,
    ) -> str:
        # Construct and return the model parameters description

        description = (
            f"{NAME_PREFIXES['model']}"
            f"{KV_SEP}"
            f"{self.short_model_name}"
            f"{ITEM_SEP}"
            f"{NAME_PREFIXES['task_type']}"
            f"{KV_SEP}"
            f"{self.task_type}"
        )

        return description

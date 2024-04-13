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

import pathlib

from pydantic import Field

from topollm.config_classes.ConfigBaseModel import ConfigBaseModel
from topollm.config_classes.constants import ITEM_SEP, KV_SEP, NAME_PREFIXES
from topollm.config_classes.enums import LMmode
from topollm.config_classes.finetuning.TokenizerModifierConfig import (
    TokenizerModifierConfig,
)


class LanguageModelConfig(ConfigBaseModel):
    lm_mode: LMmode = Field(
        ...,
        title="Language model mode.",
        description="The language model mode.",
    )

    masking_mode: str = Field(
        ...,
        title="Masking mode.",
        description="The masking mode.",
    )

    pretrained_model_name_or_path: str | pathlib.Path = Field(
        ...,
        title="Model identifier for huggingface transformers model.",
        description=f"The model identifier for the huggingface transformers model "
        f"to use for computing embeddings.",
    )

    short_model_name: str = Field(
        ...,
        title="Short model name.",
        description="The short model name.",
    )

    tokenizer_modifier: TokenizerModifierConfig = Field(
        ...,
        description="The configuration for modifying the tokenizer.",
    )

    @property
    def lanugage_model_config_description(
        self,
    ) -> str:
        # Construct and return the model parameters description

        desc = (
            f"{NAME_PREFIXES['model']}"
            f"{KV_SEP}"
            f"{self.short_model_name}"
            f"{ITEM_SEP}"
            f"{NAME_PREFIXES['masking_mode']}"
            f"{KV_SEP}"
            f"{self.masking_mode}"
        )

        return desc

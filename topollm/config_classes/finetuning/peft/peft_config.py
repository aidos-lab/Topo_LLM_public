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

"""Configuration class for the PEFT model."""

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel
from topollm.typing.enums import FinetuningMode


class PEFTConfig(ConfigBaseModel):
    """Configurations for the PEFT model."""

    finetuning_mode: FinetuningMode = Field(
        default=FinetuningMode.STANDARD,
        description="The finetuning mode of the PEFT model.",
    )

    r: int = Field(
        default=8,
        description="The r (rank) parameter of the PEFT model for LoRA.",
    )

    lora_alpha: int = Field(
        default=32,
        description="The alpha parameter of the PEFT model for LoRA.",
    )

    target_modules: list[str] | str | None = Field(
        default=[
            "query",
            "key",
            "value",
        ],
        description="The target modules of the PEFT model for LoRA.",
    )

    lora_dropout: float = Field(
        default=0.01,
        description="The dropout rate of the PEFT model for LoRA.",
    )

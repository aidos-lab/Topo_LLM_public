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

"""Configuration class for the trainer modifier."""

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel
from topollm.typing.enums import TrainerModifierMode


class TrainerModifierConfig(ConfigBaseModel):
    """Configurations for the trainer modifier."""

    mode: TrainerModifierMode = Field(
        default=TrainerModifierMode.DO_NOTHING,
        description="The trainer modifier mode.",
    )

    num_samples: int = Field(
        default=10,
        description="The number of samples to use when logging predictions.",
    )

    frequency: int = Field(
        default=400,
        description="The frequency of logging predictions.",
    )

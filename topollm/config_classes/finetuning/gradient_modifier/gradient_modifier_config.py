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

"""Configuration class for the gradient modifier."""

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel
from topollm.config_classes.constants import ITEM_SEP, KV_SEP, NAME_PREFIXES
from topollm.path_management.convert_list_to_path_part import convert_list_to_path_part
from topollm.typing.enums import GradientModifierMode


class GradientModifierConfig(ConfigBaseModel):
    """Configurations for the gradient modifier."""

    mode: GradientModifierMode = Field(
        default=GradientModifierMode.DO_NOTHING,
        description="The gradient modifier mode.",
    )

    target_modules_to_freeze: list[str] = Field(
        default_factory=list,
        description="The target modules to freeze.",
    )

    @property
    def gradient_modifier_description(self) -> str:
        """Return a description of the gradient modifier which can be used in file paths."""
        description = (
            f"{NAME_PREFIXES['GradientModifierMode']}{KV_SEP}{str(self.mode)}"
            f"{ITEM_SEP}"
            f"{NAME_PREFIXES['target_modules_to_freeze']}"
            f"{KV_SEP}"
            f"{target_modules_to_freeze_to_path_part(self.target_modules_to_freeze)}"
        )

        return description


def target_modules_to_freeze_to_path_part(
    target_modules_to_freeze: list[str],
) -> str:
    """Convert the target_modules_to_freeze to a path part."""
    return convert_list_to_path_part(
        input_list=target_modules_to_freeze,
    )

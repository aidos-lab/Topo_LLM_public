# Copyright 2024-2025
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (mail@ruppik.net)
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

"""Configuration for filtering data."""

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel
from topollm.config_classes.constants import KV_SEP, NAME_PREFIXES
from topollm.typing.enums import DescriptionType


class DataFilteringConfig(ConfigBaseModel):
    """Configuration for filtering data."""

    remove_empty_sequences: bool = Field(
        default=True,
        title="Remove empty sequences.",
        description="Remove empty sequences.",
    )

    def get_config_description(
        self,
        description_type: DescriptionType = DescriptionType.LONG,
        short_description_separator: str = "-",
    ) -> str:
        """Return the config description."""
        match description_type:
            case DescriptionType.LONG:
                description: str = (
                    f"{NAME_PREFIXES['data_filtering_remove_empty_sequences']}{KV_SEP}{self.remove_empty_sequences}"
                )
            case DescriptionType.SHORT:
                description = (
                    f"{NAME_PREFIXES['data_filtering_remove_empty_sequences']}{short_description_separator}"
                    f"{self.remove_empty_sequences}"
                )
            case _:
                msg: str = f"Unknown {description_type = }"
                raise ValueError(
                    msg,
                )

        return description

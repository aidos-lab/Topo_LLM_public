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

"""Configuration class for specifying data split."""

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel
from topollm.config_classes.constants import ITEM_SEP, KV_SEP, NAME_PREFIXES
from topollm.typing.enums import DataSplitMode


class Proportions(ConfigBaseModel):
    """Configuration for specifying proportions."""

    train: float = Field(
        default=0.8,
        title="Train proportion.",
        description="The proportion of the data to use for training.",
    )

    validation: float = Field(
        default=0.1,
        title="Validation proportion.",
        description="The proportion of the data to use for validation.",
    )

    test: float = Field(
        default=0.1,
        title="Test proportion.",
        description="The proportion of the data to use for testing.",
    )

    @property
    def config_description(
        self,
    ) -> str:
        """Return a description of the configuration."""
        description: str = (
            f"{NAME_PREFIXES['train_short']}{KV_SEP}{self.train}"
            f"{ITEM_SEP}{NAME_PREFIXES['validation_short']}{KV_SEP}{self.validation}"
            f"{ITEM_SEP}{NAME_PREFIXES['test_short']}{KV_SEP}{self.test}"
        )
        return description


class DataSplittingConfig(ConfigBaseModel):
    """Configuration for specifying data splitting."""

    data_splitting_mode: DataSplitMode = Field(
        default=DataSplitMode.DO_NOTHING,
        title="Data splitting mode.",
        description="The mode to use for splitting the data.",
    )

    split_shuffle: bool = Field(
        default=False,
        title="Split shuffle.",
        description="Whether to shuffle the data before splitting it.",
    )

    split_seed: int | None = Field(
        default=None,
        title="Split seed.",
        description="The seed to use for splitting the data.",
    )

    proportions: Proportions = Field(
        default_factory=Proportions,
        title="Proportions.",
        description="The proportions to use for splitting the data.",
    )

    @property
    def config_description(
        self,
    ) -> str:
        """Return a description of the configuration."""
        match self.data_splitting_mode:
            case DataSplitMode.DO_NOTHING:
                # No additional information needed for DO_NOTHING
                description: str = f"{NAME_PREFIXES['data_splitting_mode']}{KV_SEP}{self.data_splitting_mode}"
            case DataSplitMode.PROPORTIONS:
                description: str = f"{NAME_PREFIXES['data_splitting_mode']}{KV_SEP}{self.data_splitting_mode}"
                description += f"{ITEM_SEP}{NAME_PREFIXES['split_shuffle']}{KV_SEP}{self.split_shuffle}"
                description += f"{ITEM_SEP}{NAME_PREFIXES['split_seed']}{KV_SEP}{self.split_seed}"
                description += f"{ITEM_SEP}{self.proportions.config_description}"
            case _:
                msg: str = f"Invalid {self.data_splitting_mode = }"
                raise ValueError(msg)

        return description

# Copyright 2024-2025
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

"""Configuration class for specifying data split."""

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel
from topollm.config_classes.constants import ITEM_SEP, KV_SEP, NAME_PREFIXES
from topollm.typing.enums import DataSplitMode, DescriptionType


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

    def get_config_description(
        self,
        description_type: DescriptionType = DescriptionType.LONG,
        short_description_separator: str = "-",
    ) -> str:
        """Return a description of the configuration."""
        match description_type:
            case DescriptionType.LONG:
                description: str = (
                    f"{NAME_PREFIXES['train_short']}{KV_SEP}{self.train}"
                    f"{ITEM_SEP}{NAME_PREFIXES['validation_short']}{KV_SEP}{self.validation}"
                    f"{ITEM_SEP}{NAME_PREFIXES['test_short']}{KV_SEP}{self.test}"
                )
            case DescriptionType.SHORT:
                description: str = (
                    f"{self.train}"
                    f"{short_description_separator}{self.validation}"
                    f"{short_description_separator}{self.test}"
                )
            case _:
                msg: str = f"Invalid {description_type = }"
                raise ValueError(msg)

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

    def get_config_description(
        self,
        description_type: DescriptionType = DescriptionType.LONG,
        short_description_separator: str = "-",
    ) -> str:
        """Return a description of the configuration."""
        match self.data_splitting_mode:
            case DataSplitMode.DO_NOTHING:
                match description_type:
                    case DescriptionType.LONG:
                        description: str = f"{NAME_PREFIXES['data_splitting_mode']}{KV_SEP}{self.data_splitting_mode}"
                    case DescriptionType.SHORT:
                        description: str = f"{self.data_splitting_mode}"
                    case _:
                        msg: str = f"Invalid {description_type = }"
                        raise ValueError(msg)
                # No additional information except data_splitting_mode needed for DO_NOTHING
            case DataSplitMode.PROPORTIONS:
                match description_type:
                    case DescriptionType.LONG:
                        description: str = f"{NAME_PREFIXES['data_splitting_mode']}{KV_SEP}{self.data_splitting_mode}"
                        description += f"{ITEM_SEP}{NAME_PREFIXES['split_shuffle']}{KV_SEP}{self.split_shuffle}"
                        description += f"{ITEM_SEP}{NAME_PREFIXES['split_seed']}{KV_SEP}{self.split_seed}"
                        description += f"{ITEM_SEP}{
                            self.proportions.get_config_description(
                                description_type=description_type,
                                short_description_separator=short_description_separator,
                            )
                        }"
                    case DescriptionType.SHORT:
                        description: str = f"{self.data_splitting_mode}"
                        description += f"{short_description_separator}{self.split_shuffle}"
                        description += f"{short_description_separator}{self.split_seed}"
                        description += f"{short_description_separator}{
                            self.proportions.get_config_description(
                                description_type=description_type,
                                short_description_separator=short_description_separator,
                            )
                        }"
                    case _:
                        msg: str = f"Invalid {description_type = }"
                        raise ValueError(msg)
            case _:
                msg: str = f"Invalid {self.data_splitting_mode = }"
                raise ValueError(msg)

        return description

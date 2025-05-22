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

"""Configuration class for specifying data subsampling."""

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel
from topollm.config_classes.constants import ITEM_SEP, KV_SEP, NAME_PREFIXES
from topollm.typing.enums import DataSamplingMode, Split


class DataSubsamplingConfig(ConfigBaseModel):
    """Configuration for specifying data subsampling."""

    number_of_samples: int = Field(
        default=5_000,
        title="Number of samples from the dataset to use.",
        description="The number of samples from the dataset to use.",
    )

    sampling_mode: DataSamplingMode = Field(
        default=DataSamplingMode.TAKE_FIRST,
        title="Data sampling mode.",
        description="The mode to use for sampling from the data.",
    )

    sampling_seed: int | None = Field(
        default=987,
        title="Sampling seed.",
        description="The seed to use for sampling from the data.",
    )

    split: Split = Field(
        default=Split.TRAIN,
        title="Data split.",
        description="The data split.",
    )

    def get_short_config_description(
        self,
        short_description_separator: str = "-",
    ) -> str:
        """Return a short description of the configuration."""
        description: str = (
            f"{self.split}"
            f"{short_description_separator}"
            f"{self.number_of_samples}"
            f"{short_description_separator}"
            f"{self.sampling_mode}"
            f"{short_description_separator}"
            f"{self.sampling_seed}"
        )
        return description

    @property
    def config_description(
        self,
    ) -> str:
        """Return a description of the configuration."""
        description: str = (
            f"{NAME_PREFIXES['split']}"
            f"{KV_SEP}"
            f"{self.split}"
            f"{ITEM_SEP}"
            f"{NAME_PREFIXES['number_of_samples']}"
            f"{KV_SEP}"
            f"{self.number_of_samples}"
        )
        description += ITEM_SEP

        match self.sampling_mode:
            case DataSamplingMode.TAKE_FIRST:
                description += f"{NAME_PREFIXES['sampling_mode']}{KV_SEP}{self.sampling_mode}"
                # No additional information needed for TAKE_FIRST
            case DataSamplingMode.RANDOM:
                description += f"{NAME_PREFIXES['sampling_mode']}{KV_SEP}{self.sampling_mode}"
                description += ITEM_SEP
                description += f"{NAME_PREFIXES['sampling_seed']}{KV_SEP}{self.sampling_seed}"
            case _:
                msg: str = f"Invalid {self.sampling_mode = }"
                raise ValueError(msg)

        return description

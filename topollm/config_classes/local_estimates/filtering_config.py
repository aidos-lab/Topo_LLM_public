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

"""Configurations for specifying filtering of the data for local estimates computation."""

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel
from topollm.config_classes.constants import ITEM_SEP, KV_SEP, NAME_PREFIXES
from topollm.typing.enums import DeduplicationMode, ZeroVectorHandlingMode


class LocalEstimatesFilteringConfig(ConfigBaseModel):
    """Configurations for specifying filtering of the data for local estimates computation."""

    num_samples: int = Field(
        default=2_500,
        title="Number of samples.",
        description="The number of samples to compute the estimates for.",
    )

    zero_vector_handling_mode: ZeroVectorHandlingMode = Field(
        default=ZeroVectorHandlingMode.KEEP,
        title="Zero vector handling mode.",
        description="The mode to handle zero vectors.",
    )

    deduplication_mode: DeduplicationMode = Field(
        default=DeduplicationMode.IDENTITY,
        title="Deduplication mode.",
        description="How to handle duplicate vectors.",
    )

    @property
    def config_description(
        self,
    ) -> str:
        """Get the description of the config."""
        description: str = (
            f"{NAME_PREFIXES['num_samples']}"
            + KV_SEP
            + f"{str(object=self.num_samples)}"
            + ITEM_SEP
            + f"{NAME_PREFIXES['zero_vector_handling_mode']}"
            + KV_SEP
            + f"{str(object=self.zero_vector_handling_mode)}"
            + ITEM_SEP
            + f"{NAME_PREFIXES['deduplication_mode']}"
            + KV_SEP
            + f"{str(object=self.deduplication_mode)}"
        )

        return description

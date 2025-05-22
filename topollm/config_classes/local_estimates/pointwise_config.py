# Copyright 2024
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

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel
from topollm.config_classes.constants import ITEM_SEP, KV_SEP, NAME_PREFIXES
from topollm.typing.enums import NNeighborsMode


class LocalEstimatesPointwiseConfig(ConfigBaseModel):
    """Configurations for specifying parameters of the pointwise local estimates computation."""

    n_neighbors_mode: NNeighborsMode = Field(
        default=NNeighborsMode.RELATIVE_SIZE,
        title="Mode for specifying the number of neighbors.",
        description="The mode for specifying the number of neighbors in the pointwise estimate computation.",
    )

    absolute_n_neighbors: int = Field(
        default=128,
        title="Absolute number of neighbors.",
        description="The absolute number of neighbors to use for the pointwise estimate computation.",
    )

    relative_n_neighbors: float = Field(
        default=0.8,
        title="Relative number of neighbors.",
        description="The relative number of neighbors to use for the pointwise estimate computation.",
    )

    n_jobs: int = Field(
        default=1,
        title="Number of jobs.",
        description="The number of jobs to use for the computation.",
    )

    @property
    def config_description(
        self,
    ) -> str:
        """Get the description of the config."""
        description: str = f"{NAME_PREFIXES['n_neighbors_mode']}{KV_SEP}{str(object=self.n_neighbors_mode)}"
        description += ITEM_SEP

        if self.n_neighbors_mode == NNeighborsMode.ABSOLUTE_SIZE:
            description += f"{NAME_PREFIXES['n_neighbors']}{KV_SEP}{str(object=self.absolute_n_neighbors)}"
        elif self.n_neighbors_mode == NNeighborsMode.RELATIVE_SIZE:
            description += f"{NAME_PREFIXES['n_neighbors']}{KV_SEP}{str(object=self.relative_n_neighbors)}"
        else:
            msg: str = f"Invalid {self.n_neighbors_mode = }"
            raise ValueError(msg)

        return description

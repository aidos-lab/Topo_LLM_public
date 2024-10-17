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

"""Configuration class for embedding data preparation."""

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel
from topollm.config_classes.constants import ITEM_SEP, KV_SEP, NAME_PREFIXES
from topollm.config_classes.local_estimates.filtering_config import LocalEstimatesFilteringConfig
from topollm.config_classes.local_estimates.plot_config import LocalEstminatesPlotConfig
from topollm.config_classes.local_estimates.pointwise_config import LocalEstimatesPointwiseConfig


class LocalEstimatesConfig(ConfigBaseModel):
    """Configurations for specifying parameters of the local estimates computation."""

    method_description: str = Field(
        default="twonn",
        title="Description of the local estimates.",
        description="A description of the local estimates.",
    )

    filtering: LocalEstimatesFilteringConfig = Field(
        default_factory=LocalEstimatesFilteringConfig,
        title="Filtering configurations.",
        description="Configurations for specifying filtering of the data for local estimates computation.",
    )

    pointwise: LocalEstimatesPointwiseConfig = Field(
        default_factory=LocalEstimatesPointwiseConfig,
        title="Pointwise configurations.",
        description="Configurations for specifying parameters of the pointwise local estimates computation",
    )

    compute_global_estimates: bool = Field(
        default=True,
        title="Compute global estimates.",
        description="Whether to compute global estimates.",
    )

    plot: LocalEstminatesPlotConfig = Field(
        default_factory=LocalEstminatesPlotConfig,
        title="Plot configurations.",
        description="Configurations for specifying parameters of the local estimates plot.",
    )

    @property
    def config_description(
        self,
    ) -> str:
        """Get the description of the config."""
        description = (
            f"{NAME_PREFIXES['description']}{KV_SEP}{str(object=self.method_description)}"
            + ITEM_SEP
            + self.filtering.config_description
        )

        return description

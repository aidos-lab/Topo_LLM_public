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
from topollm.config_classes.constants import KV_SEP, NAME_PREFIXES


class FilterTokensConfig(ConfigBaseModel):
    """Configurations for filtering tokens."""

    remove_bos_token: bool = Field(
        default=False,
        title="Remove beginning of sequence token.",
        description="Whether to remove the beginning of sequence token.",
    )

    remove_eos_token: bool = Field(
        default=True,
        title="Remove end of sequence token.",
        description="Whether to remove the end of sequence token.",
    )

    remove_pad_token: bool = Field(
        default=True,
        title="Remove padding token.",
        description="Whether to remove the padding token.",
    )


class EmbeddingsDataPrepConfig(ConfigBaseModel):
    """Configurations for specifying data preparation."""

    num_samples: int = Field(
        default=30_000,
        title="Number of samples.",
        description="The number of samples to be extracted.",
    )

    filter_tokens: FilterTokensConfig = Field(
        default=FilterTokensConfig(),
        title="Filter tokens.",
        description="Configurations for filtering tokens.",
    )

    # Note: We use a feature flag in a different config group to enable or disable saving of the metadata sentences.

    @property
    def config_description(
        self,
    ) -> str:
        """Get the description of the config."""
        desc = f"{NAME_PREFIXES['num_samples']}{KV_SEP}{str(self.num_samples)}"

        return desc

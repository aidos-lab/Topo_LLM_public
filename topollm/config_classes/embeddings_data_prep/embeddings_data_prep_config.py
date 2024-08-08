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
from topollm.config_classes.embeddings_data_prep.filter_tokens_config import FilterTokensConfig
from topollm.typing.enums import EmbeddingsDataPrepSamplingMode


class EmbeddingsDataPrepSamplingConfig(ConfigBaseModel):
    """Configurations for specifying sampling in the embeddings data prep step."""

    num_samples: int = Field(
        default=30_000,
        title="Number of samples.",
        description="The number of samples to be extracted."
        "Choose size of a sample which is used to take subsets for a point-wise computation of local estimators.",
    )

    sampling_mode: EmbeddingsDataPrepSamplingMode = Field(
        default=EmbeddingsDataPrepSamplingMode.RANDOM,
        title="Sampling mode.",
        description="The sampling mode to be used.",
    )

    seed: int = Field(
        default=42,
        title="Seed.",
        description="The seed for the random number generator.",
    )

    @property
    def config_description(
        self,
    ) -> str:
        """Get the description of the config."""
        desc = (
            f"{NAME_PREFIXES['sampling_mode']}{KV_SEP}{self.sampling_mode}"
            f"{ITEM_SEP}"
            f"{NAME_PREFIXES['seed']}{KV_SEP}{self.seed!s}"
            f"{ITEM_SEP}"
            f"{NAME_PREFIXES['num_samples']}{KV_SEP}{self.num_samples!s}"
        )

        return desc


class EmbeddingsDataPrepConfig(ConfigBaseModel):
    """Configurations for specifying data preparation."""

    sampling: EmbeddingsDataPrepSamplingConfig = Field(
        default=EmbeddingsDataPrepSamplingConfig(),
        title="Sampling configurations.",
        description="Configurations for specifying sampling.",
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
        desc = f"{self.sampling.config_description}"

        return desc

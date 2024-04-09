# coding=utf-8
#
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

from pydantic import Field

from topollm.config_classes.ConfigBaseModel import ConfigBaseModel
from topollm.config_classes.constants import ITEM_SEP, KV_SEP, NAME_PREFIXES
from topollm.config_classes.data.DatasetMapConfig import DatasetMapConfig
from topollm.config_classes.embeddings.EmbeddingExtractionConfig import (
    EmbeddingExtractionConfig,
)
from topollm.config_classes.enums import Level


class EmbeddingsConfig(ConfigBaseModel):
    """Configurations for specifying embeddings."""

    dataset_map: DatasetMapConfig = Field(
        ...,
        title="Dataset map configuration.",
        description="The configuration for specifying dataset map.",
    )

    batch_size: int = Field(
        ...,
        title="Batch size for computing embeddings.",
        description="The batch size for computing embeddings.",
    )

    embedding_extraction: EmbeddingExtractionConfig = Field(
        ...,
        title="Embedding extraction configuration.",
        description="The configuration for specifying embedding extraction.",
    )

    level: Level = Field(
        default=Level.TOKEN,
        title="Level to use for computing embeddings.",
        description="The level to use for computing embeddings.",
    )

    num_workers: int = Field(
        ...,
        title="Number of workers for dataloader.",
        description=f"The number of workers for dataloader. "
        f"Note that is appears to be necessary "
        f"to set this to 0 for the 'mps' backend to work.",
    )

    @property
    def embeddings_config_description(
        self,
    ) -> str:
        desc = f"{NAME_PREFIXES['level']}"
        desc += f"{KV_SEP}"
        desc += f"{self.level}"

        return desc

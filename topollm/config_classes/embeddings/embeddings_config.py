# Copyright 2024
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

"""Configuration class for specifying embeddings."""

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel
from topollm.config_classes.constants import ITEM_SEP, KV_SEP, NAME_PREFIXES
from topollm.config_classes.data.dataset_map_config import DatasetMapConfig
from topollm.config_classes.embeddings.embedding_extraction_config import (
    EmbeddingExtractionConfig,
)
from topollm.typing.enums import EmbeddingDataHandlerMode, Level


class EmbeddingDataHandlerConfig(ConfigBaseModel):
    """Configuration for specifying the embedding data handler."""

    level: Level = Field(
        default=Level.TOKEN,
        title="Level to use for computing embeddings.",
        description="The level to use for computing embeddings.",
    )

    mode: EmbeddingDataHandlerMode = Field(
        default=EmbeddingDataHandlerMode.REGULAR,
        title="Mode for embedding data handler.",
        description="The mode for embedding data handler.",
    )

    @property
    def config_description(
        self,
    ) -> str:
        description: str = ""
        description += f"{NAME_PREFIXES['embedding_data_handler_mode']}"
        description += KV_SEP
        description += f"{str(object=self.mode)}"
        description += ITEM_SEP
        description += f"{NAME_PREFIXES['level']}"
        description += KV_SEP
        description += f"{str(object=self.level)}"

        return description


class EmbeddingsConfig(ConfigBaseModel):
    """Configurations for specifying embeddings."""

    dataset_map: DatasetMapConfig = Field(
        default_factory=DatasetMapConfig,
        title="Dataset map configuration.",
        description="The configuration for specifying dataset map.",
    )

    batch_size: int = Field(
        default=32,
        title="Batch size for computing embeddings.",
        description="The batch size for computing embeddings.",
    )

    embedding_data_handler: EmbeddingDataHandlerConfig = Field(
        default_factory=EmbeddingDataHandlerConfig,
        title="Embedding data handler configuration.",
        description="The configuration for specifying embedding data handler.",
    )

    embedding_extraction: EmbeddingExtractionConfig = Field(
        default_factory=EmbeddingExtractionConfig,
        title="Embedding extraction configuration.",
        description="The configuration for specifying embedding extraction.",
    )

    num_workers: int = Field(
        default=0,
        title="Number of workers for dataloader.",
        description=(
            "The number of workers for dataloader. "
            "Note that is appears to be necessary "
            "to set num_workers=0 for the 'mps' backend to work."
        ),
    )

    @property
    def config_description(
        self,
    ) -> str:
        # Note: The embedding extration is not part of this description and is handled separately.
        description: str = self.embedding_data_handler.config_description

        return description

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

"""Configuration class for specifying storage."""

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel
from topollm.typing.enums import ArrayStorageType, MetadataStorageType


class StorageConfig(ConfigBaseModel):
    """Configurations for specifying storage."""

    array_storage_type: ArrayStorageType = Field(
        ...,
        title="Array storage type.",
        description="The storage type for arrays.",
    )

    metadata_storage_type: MetadataStorageType = Field(
        ...,
        title="Metadata storage type.",
        description="The storage type for metadata.",
    )

    chunk_size: int = Field(
        ...,
        title="Chunk size for storage.",
        description="The chunk size for storage.",
    )

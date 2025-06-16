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


"""Configuration class for specifying storage."""

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel
from topollm.typing.enums import ArrayStorageType, MetadataStorageType


class StorageConfig(ConfigBaseModel):
    """Configurations for specifying storage."""

    array_storage_type: ArrayStorageType = Field(
        default=ArrayStorageType.ZARR,
        title="Array storage type.",
        description="The storage type for arrays.",
    )

    metadata_storage_type: MetadataStorageType = Field(
        default=MetadataStorageType.PICKLE,
        title="Metadata storage type.",
        description="The storage type for metadata.",
    )

    chunk_size: int = Field(
        default=512,
        title="Chunk size for storage.",
        description="The chunk size for storage.",
    )

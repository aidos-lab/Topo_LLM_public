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

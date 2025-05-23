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

import logging
from dataclasses import dataclass
from os import PathLike

from topollm.storage.array_storage import chunked_array_storage_zarr
from topollm.storage.array_storage.protocol import (
    ChunkedArrayStorageProtocol,
)
from topollm.storage.metadata_storage import (
    ChunkedMetadataStoragePickle,
)
from topollm.storage.metadata_storage.protocol import (
    ChunkedMetadataStorageProtocol,
)
from topollm.storage.StorageDataclasses import (
    ArrayProperties,
)
from topollm.typing.enums import ArrayStorageType, MetadataStorageType

default_logger = logging.getLogger(__name__)


@dataclass
class StoragePaths:
    """Container for paths to the storage directories."""

    array_dir: PathLike
    metadata_dir: PathLike


@dataclass
class StorageSpecification:
    """Container for storage specifications."""

    array_storage_type: ArrayStorageType
    metadata_storage_type: MetadataStorageType
    array_properties: ArrayProperties
    storage_paths: StoragePaths


class StorageFactory:
    """Factory for creating storage instances based on the storage type."""

    def __init__(
        self,
        storage_specification: StorageSpecification,
        logger: logging.Logger = default_logger,
    ) -> None:
        self.storage_specification = storage_specification
        self.logger = logger

    def get_array_storage(
        self,
    ) -> ChunkedArrayStorageProtocol:
        """Instantiate array storage backends based on the storage type."""
        if self.storage_specification.array_storage_type == ArrayStorageType.ZARR:
            storage_backend = chunked_array_storage_zarr.ChunkedArrayStorageZarr(
                array_properties=self.storage_specification.array_properties,
                root_storage_path=self.storage_specification.storage_paths.array_dir,
            )
        # Extendable to other storage types
        else:
            msg = f"Unsupported {self.storage_specification.array_storage_type = }"
            raise ValueError(msg)

        return storage_backend

    def get_metadata_storage(
        self,
    ) -> ChunkedMetadataStorageProtocol:
        match self.storage_specification.metadata_storage_type:
            case MetadataStorageType.PICKLE:
                storage_backend = ChunkedMetadataStoragePickle.ChunkedMetadataStoragePickle(
                    root_storage_path=self.storage_specification.storage_paths.metadata_dir,
                    logger=self.logger,
                )
            case _:
                msg = f"Unsupported {self.storage_specification.metadata_storage_type = }"
                raise ValueError(msg)

        return storage_backend

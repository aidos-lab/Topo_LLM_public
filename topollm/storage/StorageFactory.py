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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Imports

# Standard library imports
import logging

# Local imports
from dataclasses import dataclass
from os import PathLike

# Third party imports
from topollm.config_classes.enums import ArrayStorageType, MetadataStorageType
from topollm.storage import PickleChunkedMetadataStorage
from topollm.storage import XarrayChunkedMetadataStorage
from topollm.storage import ZarrChunkedArrayStorage
from topollm.storage.StorageProtocols import (
    ArrayProperties,
    ChunkedArrayStorageProtocol,
    ChunkedMetadataStorageProtocol,
)


# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@dataclass
class StoragePaths:
    array_dir: PathLike
    metadata_dir: PathLike


@dataclass
class StorageSpecification:
    array_storage_type: ArrayStorageType
    metadata_storage_type: MetadataStorageType
    array_properties: ArrayProperties
    storage_paths: StoragePaths


class StorageFactory:
    def __init__(
        self,
        storage_specification: StorageSpecification,
        logger: logging.Logger = logging.getLogger(__name__),
    ) -> None:
        self.storage_specification = storage_specification
        self.logger = logger

        return None

    def get_array_storage(
        self,
    ) -> ChunkedArrayStorageProtocol:
        """
        Factory function to instantiate array storage backends
        based on the storage type.
        """
        if self.storage_specification.array_storage_type == ArrayStorageType.ZARR:
            storage_backend = ZarrChunkedArrayStorage.ZarrChunkedArrayStorage(
                array_properties=self.storage_specification.array_properties,
                root_storage_path=self.storage_specification.storage_paths.array_dir,
            )
            return storage_backend
        # Extendable to other storage types
        # elif storage_type == "hdf5":
        #     return Hdf5EmbeddingStorage(store_dir)
        else:
            raise ValueError(
                f"Unsupported " f"{self.storage_specification.array_storage_type = }"
            )

    def get_metadata_storage(
        self,
    ) -> ChunkedMetadataStorageProtocol:
        if (
            self.storage_specification.metadata_storage_type
            == MetadataStorageType.XARRAY
        ):
            storage_backend = XarrayChunkedMetadataStorage.XarrayChunkedMetadataStorage(
                array_properties=self.storage_specification.array_properties,
                root_storage_path=self.storage_specification.storage_paths.metadata_dir,
                logger=self.logger,
            )
        elif (
            self.storage_specification.metadata_storage_type
            == MetadataStorageType.PICKLE
        ):
            storage_backend = PickleChunkedMetadataStorage.PickleChunkedMetadataStorage(
                root_storage_path=self.storage_specification.storage_paths.metadata_dir,
                logger=self.logger,
            )
        # Extendable to other storage types
        # elif storage_type == "pandas":
        #     return PandasMetadataStorage(store_dir)
        else:
            raise ValueError(
                f"Unsupported " f"{self.storage_specification.metadata_storage_type = }"
            )

        return storage_backend

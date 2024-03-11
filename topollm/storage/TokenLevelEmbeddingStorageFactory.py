# coding=utf-8
#
# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Julius von Rohrscheidt (julius.rohrscheidt@helmholtz-muenchen.de)
# Benjamin Ruppik (ruppik@hhu.de)
#
# This code was generated with the help of AI writing assistants
# including GitHub Copilot, ChatGPT, Bing Chat.
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

# Third party imports

# Local imports
from dataclasses import dataclass
from os import PathLike
from topollm.config_classes.enums import StorageType
from topollm.storage.StorageProtocols import (
    ChunkedArrayStorageProtocol,
    ChunkedMetadataStorageProtocol,
)
from topollm.storage.ZarrChunkedArrayStorage import ZarrChunkedArrayStorage

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@dataclass
class ArrayProperties:
    shape: tuple[int, ...]
    dtype: str  # e.g. "float32"
    chunks: tuple[int, ...]


@dataclass
class StoragePaths:
    array_dir: PathLike
    metadata_dir: PathLike


# TODO: Make this into a factory class which
# TODO: It should get the paths and create the storage backends


def get_token_level_embedding_storage(
    storage_type: StorageType,
    array_properties: ArrayProperties,
    storage_paths: StoragePaths,
) -> tuple[ChunkedArrayStorageProtocol, ChunkedMetadataStorageProtocol,]:
    """Factory function to instantiate storage backends based on the storage type.

    Args:
        storage_type:
            The type of storage to use.
        store_dir:
            The directory to store the embeddings in.

    Returns:
        An instance of a storage backend.
    """
    if storage_type == StorageType.ZARR_VECTORS_XARRAY_METADATA:
        storage_backend = ZarrChunkedArrayStorage(
            array_properties=array_properties,
            storage_paths=storage_paths,
        )
        return storage_backend
    # Extendable to other storage types
    # elif storage_type == "hdf5":
    #     return Hdf5EmbeddingStorage(store_dir)
    else:
        raise ValueError(f"Unsupported {storage_type = }")

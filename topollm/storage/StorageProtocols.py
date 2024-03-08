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
import numpy as np
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from os import PathLike


# Third party imports

# Local imports
from topollm.config_classes.enums import StorageType
from topollm.storage.ZarrChunkedArrayStorage import ZarrChunkedArrayStorage

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@dataclass
class ChunkIdentifier:
    chunk_idx: int
    start_idx: int


@dataclass
class ArrayDataChunk:
    """
    Dataclass to hold a single embedding chunk.
    """
    batch_of_sequences_embedding_array: np.ndarray
    chunk_identifier: ChunkIdentifier

@dataclass
class MetaDataChunk:
    """
    Dataclass to hold a single metadata chunk.
    """
    batch: dict
    chunk_identifier: ChunkIdentifier

@dataclass
class ArrayProperties:
    shape: tuple[int, ...]
    dtype: str  # e.g. "float32"
    chunks: tuple[int, ...]


@dataclass
class StoragePaths:
    array_dir: PathLike
    metadata_dir: PathLike


@runtime_checkable
class ChunkedArrayStorageProtocol(Protocol):
    def open(
        self,
    ) -> None:
        """Initializes the storage with specified configuration."""
        ...

    def write_chunk(
        self,
        data_chunk: ArrayDataChunk,
    ) -> None:
        """Writes a chunk of data starting from a specific index."""
        ...

    def read_chunk(
        self,
        chunk_identifier: ChunkIdentifier,
    ) -> ArrayDataChunk:
        """Reads a chunk of data determined by the identifier."""
        ...


def get_token_level_embedding_storage(
    storage_type: StorageType,
    array_properties: ArrayProperties,
    storage_paths: StoragePaths,
) -> ChunkedArrayStorageProtocol:
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

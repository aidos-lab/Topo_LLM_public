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
import logging
import numpy as np
import os
import warnings
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from os import PathLike


# Third party imports
import zarr

# Local imports
from topollm.config_classes.enums import StorageType

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@dataclass
class ChunkIdentifier:
    chunk_idx: int
    start_idx: int


@dataclass
class TokenLevelDataChunk:
    """
    Dataclass to hold one embedding chunk
    and the batch containing the corresponding dataset entries.
    """

    batch_of_sequences_embedding_array: np.ndarray
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
class TokenLevelEmbeddingStorageProtocol(Protocol):
    def open(
        self,
    ) -> None:
        """Initializes the storage with specified configuration."""
        ...

    def write_chunk(
        self,
        data_chunk: TokenLevelDataChunk,
    ) -> None:
        """Writes a chunk of data starting from a specific index."""
        ...

    def read_chunk(
        self,
        chunk_identifier: ChunkIdentifier,
    ) -> TokenLevelDataChunk:
        """Reads a chunk of data determined by the identifier."""
        ...


class TokenLevelZarrXarrayEmbeddingStorage(TokenLevelEmbeddingStorageProtocol):
    """
    A storage protocol backend for token level embeddings
    using Zarr and Xarray.
    """

    def __init__(
        self,
        array_properties: ArrayProperties,
        storage_paths: StoragePaths,
        logger: logging.Logger = logging.getLogger(__name__),
    ):
        self.array_properties = array_properties
        self.storage_paths = storage_paths

    def open(
        self,
    ) -> None:
        # # # #
        # Open zarr array (for embeddings)
        os.makedirs(
            self.storage_paths.array_dir,
            exist_ok=True,
        )
        self.zarr_array = zarr.open(
            store=self.storage_paths.array_dir,  # type: ignore
            mode="w",
            shape=self.array_properties.shape,
            dtype=self.array_properties.dtype,
            chunks=self.array_properties.chunks,
        )

        # # # #
        # Open xarray (for metadata)

        warnings.warn(
            message=f"xarray Not implemented yet",
        )

        # TODO 2: Continue here

        return

    def write_chunk(
        self,
        data_chunk: TokenLevelDataChunk,
    ) -> None:
        # TODO: Update this to work with the DataClass

        # TODO 1: Implement saving of the embeddings
        # TODO 2: Implement saving of the metadata

        warnings.warn(
            message=f"write_chunk Not implemented yet",
        )

        return  # TODO fake implementation

        self.zarr_array[start_idx : start_idx + len(data)] = data

    def read_chunk(
        self,
        chunk_identifier: ChunkIdentifier,
    ) -> TokenLevelDataChunk:
        # TODO implement

        raise NotImplementedError


def get_token_level_embedding_storage(
    storage_type: StorageType,
    array_properties: ArrayProperties,
    storage_paths: StoragePaths,
) -> TokenLevelEmbeddingStorageProtocol:
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
        return TokenLevelZarrXarrayEmbeddingStorage(
            array_properties=array_properties,
            storage_paths=storage_paths,
        )
    # Extendable to other storage types
    # elif storage_type == "hdf5":
    #     return Hdf5EmbeddingStorage(store_dir)
    else:
        raise ValueError(f"Unsupported {storage_type = }")

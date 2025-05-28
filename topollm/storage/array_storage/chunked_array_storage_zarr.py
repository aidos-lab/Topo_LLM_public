# Copyright 2024-2025
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


import logging
import os
import pathlib

import zarr
import zarr.creation

from topollm.storage.StorageDataclasses import (
    ArrayDataChunk,
    ArrayProperties,
    ChunkIdentifier,
)

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


class ChunkedArrayStorageZarr:
    """Storage protocol backend for chunked arrays using Zarr.

    Note: We do not need to inherit from the storage protocols,
    since we are not relying on an abstract base class.
    """

    def __init__(
        self,
        array_properties: ArrayProperties,
        root_storage_path: os.PathLike,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Initialize the storage protocol backend."""
        self.array_properties = array_properties
        self.root_storage_path = pathlib.Path(
            root_storage_path,
        )
        self.logger: logging.Logger = logger

    def open(
        self,
    ) -> None:
        # # # #
        # Open zarr array (for embeddings)
        pathlib.Path(self.root_storage_path).mkdir(
            parents=True,
            exist_ok=True,
        )

        self.zarr_array = zarr.creation.open_array(
            store=self.root_storage_path,
            mode="w",
            shape=self.array_properties.shape,
            dtype=self.array_properties.dtype,
            chunks=self.array_properties.chunks,  # type: ignore - problem with zarr chunk size typing
        )

    def write_chunk(
        self,
        data_chunk: ArrayDataChunk,
    ) -> None:
        start_idx = data_chunk.chunk_identifier.start_idx
        end_idx = data_chunk.chunk_identifier.end_idx

        data = data_chunk.batch_of_sequences_embedding_array

        self.zarr_array[start_idx:end_idx,] = data

    def read_chunk(
        self,
        chunk_identifier: ChunkIdentifier,
    ) -> ArrayDataChunk:
        start_idx = chunk_identifier.start_idx
        end_idx = chunk_identifier.end_idx

        data = self.zarr_array[start_idx:end_idx,]

        return ArrayDataChunk(
            batch_of_sequences_embedding_array=data,
            chunk_identifier=chunk_identifier,
        )

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


class ChunkedArrayStorageZarr:
    """Storage protocol backend for chunked arrays using Zarr.

    Note: We do not need to inherit from the storage protocols,
    since we are not relying on an abstract base class.
    """

    def __init__(
        self,
        array_properties: ArrayProperties,
        root_storage_path: os.PathLike,
        logger: logging.Logger = logging.getLogger(__name__),
    ):
        self.array_properties = array_properties
        self.root_storage_path = pathlib.Path(
            root_storage_path,
        )
        self.logger = logger

    def open(
        self,
    ) -> None:
        # # # #
        # Open zarr array (for embeddings)
        os.makedirs(
            self.root_storage_path,
            exist_ok=True,
        )

        self.zarr_array = zarr.creation.open_array(
            store=self.root_storage_path,
            mode="w",
            shape=self.array_properties.shape,
            dtype=self.array_properties.dtype,
            chunks=self.array_properties.chunks,  # type: ignore
        )

        return

    def write_chunk(
        self,
        data_chunk: ArrayDataChunk,
    ) -> None:
        start_idx = data_chunk.chunk_identifier.start_idx
        end_idx = data_chunk.chunk_identifier.end_idx

        data = data_chunk.batch_of_sequences_embedding_array

        self.zarr_array[start_idx:end_idx,] = data

        return

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

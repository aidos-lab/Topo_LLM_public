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
from importlib import metadata
import logging
import os
import pathlib
import pickle
import warnings

# Third party imports

# Local imports
from topollm.storage.StorageProtocols import (
    MetadataChunk,
    ChunkIdentifier,
    ArrayProperties,
)

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def chunk_identifier_str(
    chunk_identifier: ChunkIdentifier,
    fill_zeros: int = 5,
) -> str:
    return str(chunk_identifier.chunk_idx).zfill(fill_zeros)


class PickleChunkedMetadataStorage:
    """
    A storage protocol backend for embedding metadata using Pickles.

    Note: We do not need to inherit from the storage protocols,
    since we are not relying on an abstract base class.
    """

    def __init__(
        self,
        root_storage_path: os.PathLike,
        logger: logging.Logger = logging.getLogger(__name__),
    ):
        self.root_storage_path = root_storage_path
        self.storage_dir = pathlib.Path(
            self.root_storage_path,
            "pickle_chunked_metadata_storage",
        )

        self.logger = logger

        return None

    def open(
        self,
    ) -> None:
        # # # #
        # Create the folder for the storage if it does not exist
        os.makedirs(
            self.storage_dir,
            exist_ok=True,
        )

        return None

    @classmethod
    def chunk_file_name(
        cls,
        chunk_identifier: ChunkIdentifier,
    ) -> str:
        chunk_id_str = chunk_identifier_str(
            chunk_identifier=chunk_identifier,
        )

        return f"chunk_" f"{chunk_id_str}" f".pkl"

    def chunk_file_path(
        self,
        chunk_identifier: ChunkIdentifier,
    ) -> pathlib.Path:
        return pathlib.Path(
            self.storage_dir,
            self.chunk_file_name(chunk_identifier),
        )

    def write_chunk(
        self,
        data_chunk: MetadataChunk,
    ) -> None:
        chunk_file_path = self.chunk_file_path(
            chunk_identifier=data_chunk.chunk_identifier,
        )

        with open(
            file=chunk_file_path,
            mode="wb",
        ) as file:
            pickle.dump(
                obj=data_chunk.batch,
                file=file,
            )

        return None

    def read_chunk(
        self,
        chunk_identifier: ChunkIdentifier,
    ) -> MetadataChunk:
        chunk_file_path = self.chunk_file_path(
            chunk_identifier=chunk_identifier,
        )

        with open(
            file=chunk_file_path,
            mode="rb",
        ) as file:
            batch = pickle.load(
                file=file,
            )

        metadata_chunk = MetadataChunk(
            chunk_identifier=chunk_identifier,
            batch=batch,
        )

        return metadata_chunk

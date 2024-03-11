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
import os
import warnings

# Third party imports

# Local imports
from topollm.storage.StorageProtocols import (
    MetaDataChunk,
    ArrayProperties,
    ChunkIdentifier,
)

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class XarrayChunkedMetadataStorage:
    """
    A storage protocol backend for embedding metadata using Xarray.

    Note: We do not need to inherit from the storage protocols,
    since we are not relying on an abstract base class.
    """

    # TODO Implement this class

    def __init__(
        self,
        array_properties: ArrayProperties,
        storage_path: os.PathLike,
        logger: logging.Logger = logging.getLogger(__name__),
    ):
        self.array_properties = array_properties
        self.storage_path = storage_path

    def open(
        self,
    ) -> None:
        # # # #
        # Open xarray (for metadata)

        file_name = "metadata.nc"

        warnings.warn(
            message=f"xarray Not implemented yet",
        )

        # TODO 2: Continue here

        return

    def write_chunk(
        self,
        data_chunk: MetaDataChunk,
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
    ) -> MetaDataChunk:
        # TODO implement

        raise NotImplementedError

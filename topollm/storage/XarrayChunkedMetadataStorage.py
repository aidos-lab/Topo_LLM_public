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
import os
import warnings

# Third party imports

# Local imports
from topollm.storage.StorageProtocols import (
    MetaDataChunk,
    ChunkIdentifier,
    ArrayProperties,
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
        root_storage_path: os.PathLike,
        logger: logging.Logger = logging.getLogger(__name__),
    ):
        self.array_properties = array_properties
        self.root_storage_path = root_storage_path
        self.logger = logger

    def open(
        self,
    ) -> None:
        # # # #
        # Open xarray (for metadata)
        os.makedirs(
            self.root_storage_path,
            exist_ok=True,
        )

        file_name = "metadata.nc"  # TODO: Might not be neccessary

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

        return None  # TODO fake implementation

    def read_chunk(
        self,
        chunk_identifier: ChunkIdentifier,
    ) -> MetaDataChunk:
        # TODO implement

        raise NotImplementedError  # ! Remove this line

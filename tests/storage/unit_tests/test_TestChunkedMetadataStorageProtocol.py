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

# System imports
import logging
import pathlib
import pprint
from abc import ABC, abstractmethod

# Third-party imports
import numpy as np
import pytest

# Local imports
from topollm.storage import StorageProtocols
from topollm.storage.metadata_storage import ChunkedMetadataStoragePickle

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class ChunkedMetadataStorageFactory(ABC):
    """
    Abstract factory for creating storage instances.
    """

    @abstractmethod
    def create_storage(
        self,
    ) -> StorageProtocols.ChunkedMetadataStorageProtocol:
        """
        Creates and returns a storage instance, with all necessary
        arguments handled internally by the factory.
        """
        pass


class _ChunkedMetadataStorageProtocol(ABC):
    """
    Abstract base class for testing implementations of ChunkedArrayStorageProtocol.
    """

    @pytest.fixture
    @abstractmethod
    def storage_factory(
        self,
        request,
    ) -> ChunkedMetadataStorageFactory:
        """
        Should be overridden by subclasses to provide a concrete storage factory.
        The factory itself should handle all necessary arguments for storage creation.
        """
        pass

    @pytest.fixture
    def storage(
        self,
        storage_factory: ChunkedMetadataStorageFactory,
    ) -> StorageProtocols.ChunkedMetadataStorageProtocol:
        """
        Dynamic storage instance creation using the provided factory.
        """
        return storage_factory.create_storage()

    def test_write_and_read_chunk(
        self,
        storage: StorageProtocols.ChunkedMetadataStorageProtocol,
        example_batch: dict,
        chunk_idx: int,
        logger_fixture: logging.Logger,
    ) -> None:
        """
        Test that data written to storage can be read back accurately.
        """

        storage.open()

        logger_fixture.info(
            f"Testing with example_batch:\n"
            f"{pprint.pformat(example_batch)}\n"
            f"and\n{chunk_idx = }"
        )

        chunk_identifier = StorageProtocols.ChunkIdentifier(
            chunk_idx=chunk_idx,
            start_idx=-1,  # Not used for metadata
            chunk_length=-1,  # Not used for metadata
        )

        metadata_chunk = StorageProtocols.MetadataChunk(
            batch=example_batch,
            chunk_identifier=chunk_identifier,
        )

        storage.write_chunk(
            data_chunk=metadata_chunk,
        )
        read_chunk: StorageProtocols.MetadataChunk = storage.read_chunk(
            chunk_identifier=chunk_identifier,
        )

        logger_fixture.info(metadata_chunk)
        logger_fixture.info(read_chunk)

        # FIXME Implement proper comparison for metadata chunks
        #! Currently this test fails with "RuntimeError: Boolean value of Tensor with more than one value is ambiguous"
        assert read_chunk == metadata_chunk, "Read data does not match written data"


class PickleChunkedMetadataStorageFactory(ChunkedMetadataStorageFactory):
    """
    Factory for creating ZarrChunkedArrayStorage instances.
    """

    def __init__(
        self,
        tmp_path: pathlib.Path,
        logger: logging.Logger,
    ):
        self.tmp_path = tmp_path
        self.logger = logger

    def create_storage(
        self,
    ):
        storage_path = pathlib.Path(
            self.tmp_path,
            "pickle_chunked_metadata_storage_test",
        )
        self.logger.info(
            f"Creating PickleChunkedMetadataStorage storage " f"at {storage_path = }"
        )

        return ChunkedMetadataStoragePickle.ChunkedMetadataStoragePickle(
            root_storage_path=storage_path,
            logger=self.logger,
        )


class TestPickleChunkedMetadataStorage(_ChunkedMetadataStorageProtocol):
    @pytest.fixture
    def storage_factory(  # type: ignore
        self,
        request: pytest.FixtureRequest,
        test_data_dir: pathlib.Path,
        logger_fixture: logging.Logger,
    ) -> ChunkedMetadataStorageFactory:
        """
        Provides a concrete factory instance, initialized with all necessary arguments
        for creating a ZarrChunkedArrayStorage instance.
        """
        return PickleChunkedMetadataStorageFactory(
            tmp_path=test_data_dir,
            logger=logger_fixture,
        )

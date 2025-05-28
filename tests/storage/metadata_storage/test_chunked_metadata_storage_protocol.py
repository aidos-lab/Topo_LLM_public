# Copyright 2024
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
import pathlib
import pprint
from abc import ABC, abstractmethod

import pytest

import topollm.storage.metadata_storage.MetadataChunk
import topollm.storage.metadata_storage.protocol
from topollm.storage import StorageDataclasses
from topollm.storage.metadata_storage import ChunkedMetadataStoragePickle


class ChunkedMetadataStorageFactory(ABC):
    """Abstract factory for creating storage instances."""

    @abstractmethod
    def create_storage(
        self,
    ) -> topollm.storage.metadata_storage.protocol.ChunkedMetadataStorageProtocol:
        """Create and return a storage instance, with all necessary arguments handled internally by the factory."""
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
    ) -> topollm.storage.metadata_storage.protocol.ChunkedMetadataStorageProtocol:
        """
        Dynamic storage instance creation using the provided factory.
        """
        return storage_factory.create_storage()

    def test_write_and_read_chunk(
        self,
        storage: topollm.storage.metadata_storage.protocol.ChunkedMetadataStorageProtocol,
        example_batch: dict,
        chunk_idx: int,
        logger_fixture: logging.Logger,
    ) -> None:
        """
        Test that data written to storage can be read back accurately.
        """

        storage.open()

        logger_fixture.info(f"Testing with example_batch:\n{pprint.pformat(example_batch)}\nand\n{chunk_idx = }")

        chunk_identifier = StorageDataclasses.ChunkIdentifier(
            chunk_idx=chunk_idx,
            start_idx=-1,  # Not used for metadata
            chunk_length=-1,  # Not used for metadata
        )

        metadata_chunk = topollm.storage.metadata_storage.MetadataChunk.MetadataChunk(
            batch=example_batch,
            chunk_identifier=chunk_identifier,
        )

        storage.write_chunk(
            data_chunk=metadata_chunk,
        )
        read_chunk: topollm.storage.metadata_storage.MetadataChunk.MetadataChunk = storage.read_chunk(
            chunk_identifier=chunk_identifier,
        )

        logger_fixture.info(metadata_chunk)
        logger_fixture.info(read_chunk)

        # This assertion uses the __eq__ method of MetadataChunk
        assert read_chunk == metadata_chunk, "Read data does not match written data"


class ChunkedMetadataStoragePickleFactory(ChunkedMetadataStorageFactory):
    """
    Factory for creating ChunkedArrayStoragePickle instances.
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
        self.logger.info(f"Creating ChunkedMetadataStoragePickle storage at {storage_path = }")

        return ChunkedMetadataStoragePickle.ChunkedMetadataStoragePickle(
            root_storage_path=storage_path,
            logger=self.logger,
        )


class TestChunkedMetadataStoragePickle(_ChunkedMetadataStorageProtocol):
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
        return ChunkedMetadataStoragePickleFactory(
            tmp_path=test_data_dir,
            logger=logger_fixture,
        )

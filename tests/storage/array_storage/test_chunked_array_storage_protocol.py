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
import pathlib
from abc import ABC, abstractmethod

import numpy as np
import pytest

import topollm.storage.array_storage.protocol
from topollm.storage import StorageDataclasses
from topollm.storage.array_storage import chunked_array_storage_zarr


@pytest.fixture()
def array_properties(
    request: pytest.FixtureRequest,
) -> StorageDataclasses.ArrayProperties:
    """Fixture that provides ArrayProperties based on parameterized inputs."""
    shape = request.param
    return StorageDataclasses.ArrayProperties(
        shape=shape,
        dtype="float32",
        chunks=(5,),
    )


class ChunkedArrayStorageFactory(ABC):
    """Abstract factory for creating storage instances."""

    @abstractmethod
    def create_storage(
        self,
    ) -> topollm.storage.array_storage.protocol.ChunkedArrayStorageProtocol:
        """Create and return a storage instance, with all necessary arguments handled internally by the factory."""
        pass


class _ChunkedArrayStorageProtocol(ABC):
    """Abstract base class for testing implementations of ChunkedArrayStorageProtocol."""

    @pytest.fixture()
    @abstractmethod
    def storage_factory(
        self,
        request,
    ) -> ChunkedArrayStorageFactory:
        """
        Should be overridden by subclasses to provide a concrete storage factory.
        The factory itself should handle all necessary arguments for storage creation.
        """
        pass

    @pytest.fixture()
    def storage(
        self,
        storage_factory: ChunkedArrayStorageFactory,
    ) -> topollm.storage.array_storage.protocol.ChunkedArrayStorageProtocol:
        """
        Dynamic storage instance creation using the provided factory.
        """
        return storage_factory.create_storage()

    @pytest.mark.parametrize(
        "array_properties, chunk_length, start_idx",
        [
            pytest.param((10, 100), 8, 0, id="2D-size_10x100-start_0"),
            pytest.param((10, 100), 2, 8, id="2D-size_10x100-start_10"),
            pytest.param((100, 200), 10, 0, id="2D-size_100x200-start_0"),
            pytest.param((100, 200), 20, 10, id="2D-size_100x200-start_10"),
            pytest.param((800, 512, 786), 10, 0, id="3D-size_800x512x786-start_0"),
            pytest.param((800, 512, 786), 20, 10, id="3D-size_800x512x786-start_10"),
            # Add more combinations as needed
        ],
        indirect=["array_properties"],  # Specifies which parameters are for fixtures
    )
    def test_write_and_read_chunk(
        self,
        storage: topollm.storage.array_storage.protocol.ChunkedArrayStorageProtocol,
        array_properties: StorageDataclasses.ArrayProperties,
        chunk_length: int,
        start_idx: int,
    ) -> None:
        """
        Test that data written to storage can be read back accurately.
        """

        storage.open()

        # Take the length of the chunk to write from `chunk_length`
        # and the remaining dimensions from `array_properties.shape`.
        chunk_shape = (chunk_length,) + array_properties.shape[1:]  # The '+' here is tuple concatenation

        random_array = np.random.rand(*chunk_shape)
        data = random_array.astype(
            dtype=array_properties.dtype,
        )

        chunk_identifier = StorageDataclasses.ChunkIdentifier(
            chunk_idx=0,
            start_idx=start_idx,
            chunk_length=chunk_length,
        )

        data_chunk = StorageDataclasses.ArrayDataChunk(
            batch_of_sequences_embedding_array=data,
            chunk_identifier=chunk_identifier,
        )

        storage.write_chunk(
            data_chunk=data_chunk,
        )
        read_chunk = storage.read_chunk(
            chunk_identifier=chunk_identifier,
        )

        assert np.array_equal(
            read_chunk.batch_of_sequences_embedding_array,
            data,
        ), "Read data does not match written data"


class ChunkedArrayStorageZarrFactory(ChunkedArrayStorageFactory):
    """
    Factory for creating ZarrChunkedArrayStorage instances.
    """

    def __init__(
        self,
        array_properties: StorageDataclasses.ArrayProperties,
        tmp_path: pathlib.Path,
        logger: logging.Logger,
    ):
        self.array_properties = array_properties
        self.tmp_path = tmp_path
        self.logger = logger

    def create_storage(
        self,
    ):
        storage_path = pathlib.Path(
            self.tmp_path,
            "zarr_chunked_array_storage_test",
        )
        self.logger.info(f"Creating ZarrChunkedArrayStorage storage " f"at {storage_path = }")

        return chunked_array_storage_zarr.ChunkedArrayStorageZarr(
            array_properties=self.array_properties,
            root_storage_path=storage_path,
            logger=self.logger,
        )


class TestChunkedArrayStorageZarr(_ChunkedArrayStorageProtocol):
    @pytest.fixture
    def storage_factory(  # type: ignore
        self,
        request: pytest.FixtureRequest,
        array_properties: StorageDataclasses.ArrayProperties,
        test_data_dir: pathlib.Path,
        logger_fixture: logging.Logger,
    ) -> ChunkedArrayStorageFactory:
        """
        Provides a concrete factory instance, initialized with all necessary arguments
        for creating a ZarrChunkedArrayStorage instance.
        """
        return ChunkedArrayStorageZarrFactory(
            array_properties=array_properties,
            tmp_path=test_data_dir,
            logger=logger_fixture,
        )

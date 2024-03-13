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
from abc import ABC, abstractmethod

# Third-party imports
import numpy as np
import pytest

# Local imports
from topollm.storage import StorageProtocols, ZarrChunkedArrayStorage

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@pytest.fixture(scope="function")
def array_properties(
    request,
) -> StorageProtocols.ArrayProperties:
    """
    Fixture that provides ArrayProperties based on parameterized inputs.
    """
    shape = request.param
    return StorageProtocols.ArrayProperties(
        shape=shape,
        dtype="float32",
        chunks=(5,),
    )


class _ChunkedArrayStorageProtocol(ABC):
    """
    Abstract base class for testing implementations of ChunkedArrayStorageProtocol.
    """

    @pytest.fixture
    @abstractmethod
    def storage(
        self,
        **kwargs,
    ) -> StorageProtocols.ChunkedArrayStorageProtocol:
        """
        Should be overridden by subclasses for specific implementations.
        """
        raise NotImplementedError(
            "This method should be implemented by subclasses.",
        )

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
        storage: StorageProtocols.ChunkedArrayStorageProtocol,
        array_properties: StorageProtocols.ArrayProperties,
        chunk_length: int,
        start_idx: int,
    ) -> None:
        """
        Test that data written to storage can be read back accurately.
        """

        storage.open()

        # Take the length of the chunk to write from `chunk_length`
        # and the remaining dimensions from `array_properties.shape`.
        chunk_shape = (chunk_length,) + array_properties.shape[
            1:
        ]  # The '+' here is tuple concatenation

        random_array = np.random.rand(*chunk_shape)
        data = random_array.astype(
            dtype=array_properties.dtype,
        )

        chunk_identifier = StorageProtocols.ChunkIdentifier(
            chunk_idx=0,
            start_idx=start_idx,
            chunk_length=chunk_length,
        )

        data_chunk = StorageProtocols.ArrayDataChunk(
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


class TestZarrChunkedArrayStorage(_ChunkedArrayStorageProtocol):
    @pytest.fixture
    def storage(  # type: ignore
        self,
        tmp_path: pathlib.Path,
        array_properties: StorageProtocols.ArrayProperties,
        logger_fixture: logging.Logger,
    ) -> ZarrChunkedArrayStorage.ZarrChunkedArrayStorage:
        """
        Concrete fixture providing a ZarrChunkedArrayStorage instance.
        """
        storage_path = pathlib.Path(
            tmp_path,
            "zarr_test",
        )

        return ZarrChunkedArrayStorage.ZarrChunkedArrayStorage(
            array_properties=array_properties,
            storage_path=storage_path,
            logger=logger_fixture,
        )

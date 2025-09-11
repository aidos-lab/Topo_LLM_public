import logging
import os
import pathlib
import pickle

from topollm.storage.metadata_storage.MetadataChunk import MetadataChunk
from topollm.storage.StorageDataclasses import ChunkIdentifier

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def chunk_identifier_str(
    chunk_identifier: ChunkIdentifier,
    fill_zeros: int = 5,
) -> str:
    return str(chunk_identifier.chunk_idx).zfill(fill_zeros)


class ChunkedMetadataStoragePickle:
    """Storage protocol backend for embedding metadata using Pickle.

    Note: We do not need to inherit from the storage protocols,
    since we are not relying on an abstract base class.
    """

    def __init__(
        self,
        root_storage_path: os.PathLike,
        logger: logging.Logger = default_logger,
    ):
        self.root_storage_path: pathlib.Path = pathlib.Path(
            root_storage_path,
        )
        self.storage_dir = pathlib.Path(
            self.root_storage_path,
            "pickle_chunked_metadata_storage",
        )

        self.logger = logger

    def open(
        self,
    ) -> None:
        # # # #
        # Create the folder for the storage if it does not exist
        pathlib.Path(self.storage_dir).mkdir(
            parents=True,
            exist_ok=True,
        )

    @classmethod
    def chunk_file_name(
        cls,
        chunk_identifier: ChunkIdentifier,
    ) -> str:
        chunk_id_str = chunk_identifier_str(
            chunk_identifier=chunk_identifier,
        )

        return f"chunk_{chunk_id_str}.pkl"

    def chunk_file_path(
        self,
        chunk_identifier: ChunkIdentifier,
    ) -> pathlib.Path:
        return pathlib.Path(
            self.storage_dir,
            self.chunk_file_name(chunk_identifier=chunk_identifier),
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

"""Protocol for chunked metadata storage backends."""

from typing import Protocol, runtime_checkable

from topollm.storage.metadata_storage.MetadataChunk import MetadataChunk
from topollm.storage.StorageDataclasses import ChunkIdentifier


@runtime_checkable
class ChunkedMetadataStorageProtocol(Protocol):
    """Protocol for chunked metadata storage backends."""

    def open(
        self,
    ) -> None:
        """Initialize the storage with specified configuration."""
        ...  # pragma: no cover

    def write_chunk(
        self,
        data_chunk: MetadataChunk,
    ) -> None:
        """Write a chunk of data starting from a specific index."""
        ...  # pragma: no cover

    def read_chunk(
        self,
        chunk_identifier: ChunkIdentifier,
    ) -> MetadataChunk:
        """Read a chunk of data determined by the identifier."""
        ...  # pragma: no cover

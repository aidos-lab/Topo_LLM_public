"""Protocol for chunked array storage."""

from typing import Protocol, runtime_checkable

from topollm.storage.StorageDataclasses import ArrayDataChunk, ChunkIdentifier


@runtime_checkable
class ChunkedArrayStorageProtocol(Protocol):
    """Protocol for chunked array storage."""

    def open(
        self,
    ) -> None:
        """Initialize the storage with specified configuration."""
        ...  # pragma: no cover

    def write_chunk(
        self,
        data_chunk: ArrayDataChunk,
    ) -> None:
        """Write a chunk of data starting from a specific index."""
        ...  # pragma: no cover

    def read_chunk(
        self,
        chunk_identifier: ChunkIdentifier,
    ) -> ArrayDataChunk:
        """Read a chunk of data determined by the identifier."""
        ...  # pragma: no cover

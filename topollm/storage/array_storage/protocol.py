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

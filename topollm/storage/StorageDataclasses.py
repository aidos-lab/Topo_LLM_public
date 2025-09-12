from dataclasses import dataclass

import numpy as np


@dataclass
class ChunkIdentifier:
    """Dataclass to hold information about a specific chunk."""

    chunk_idx: int
    start_idx: int
    chunk_length: int

    @property
    def end_idx(
        self,
    ) -> int:
        return self.start_idx + self.chunk_length


@dataclass
class ArrayDataChunk:
    """Dataclass to hold a single embedding chunk."""

    batch_of_sequences_embedding_array: np.ndarray
    chunk_identifier: ChunkIdentifier


@dataclass
class ArrayProperties:
    """Dataclass to hold properties of the embedding array to be stored."""

    shape: tuple[int, ...]
    dtype: str  # e.g. "float32"
    chunks: tuple[int, ...]

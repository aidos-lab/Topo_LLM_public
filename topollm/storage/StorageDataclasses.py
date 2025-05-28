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


from dataclasses import dataclass

import numpy as np


@dataclass
class ChunkIdentifier:
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
    """
    Dataclass to hold a single embedding chunk.
    """

    batch_of_sequences_embedding_array: np.ndarray
    chunk_identifier: ChunkIdentifier


@dataclass
class ArrayProperties:
    shape: tuple[int, ...]
    dtype: str  # e.g. "float32"
    chunks: tuple[int, ...]

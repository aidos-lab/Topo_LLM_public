"""Dataclass to hold a single metadata chunk."""

from dataclasses import dataclass
from typing import Any

import torch

from topollm.storage.StorageDataclasses import ChunkIdentifier


@dataclass
class MetadataChunk:
    """Dataclass to hold a single metadata chunk."""

    batch: dict
    chunk_identifier: ChunkIdentifier

    def __eq__(
        self,
        other: Any,
    ) -> bool:
        """Override equality check to compare MetadataChunk instances."""
        if not isinstance(
            other,
            MetadataChunk,
        ):
            return False
        if self.chunk_identifier != other.chunk_identifier:
            return False

        # Ensure both batches have the same keys before comparing values
        if set(self.batch.keys()) != set(other.batch.keys()):
            return False

        for key, value in self.batch.items():
            other_value = other.batch[key]
            # Check if both values are tensors before comparing
            if torch.is_tensor(value) and torch.is_tensor(other_value):
                if not torch.equal(
                    value,
                    other_value,
                ):
                    return False
            elif value != other_value:
                # Fallback for non-tensor values
                return False
        return True

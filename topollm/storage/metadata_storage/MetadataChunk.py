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

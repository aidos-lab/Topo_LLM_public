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

from typing import Protocol, runtime_checkable

from topollm.storage.StorageDataclasses import ArrayDataChunk, ChunkIdentifier


@runtime_checkable
class ChunkedArrayStorageProtocol(Protocol):
    def open(
        self,
    ) -> None:
        """Initializes the storage with specified configuration."""
        ...  # pragma: no cover

    def write_chunk(
        self,
        data_chunk: ArrayDataChunk,
    ) -> None:
        """Writes a chunk of data starting from a specific index."""
        ...  # pragma: no cover

    def read_chunk(
        self,
        chunk_identifier: ChunkIdentifier,
    ) -> ArrayDataChunk:
        """Reads a chunk of data determined by the identifier."""
        ...  # pragma: no cover

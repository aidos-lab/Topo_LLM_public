# Copyright 2024-2025
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

"""Base class for all data handlers for computing and storing embeddings."""

import logging
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm
from transformers import PreTrainedModel

from topollm.compute_embeddings.embedding_extractor.protocol import EmbeddingExtractor
from topollm.compute_embeddings.move_batch_to_cpu import move_batch_to_cpu
from topollm.storage.array_storage.protocol import ChunkedArrayStorageProtocol
from topollm.storage.metadata_storage.MetadataChunk import MetadataChunk
from topollm.storage.metadata_storage.protocol import ChunkedMetadataStorageProtocol
from topollm.storage.StorageDataclasses import ArrayDataChunk, ChunkIdentifier
from topollm.typing.enums import Verbosity
from topollm.typing.types import TransformersTokenizer

default_device: torch.device = torch.device(
    device="cpu",
)
default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


class BaseEmbeddingDataHandler(ABC):
    """Base class for handling embedding computation and storage."""

    def __init__(
        self,
        array_storage_backend: ChunkedArrayStorageProtocol,
        metadata_storage_backend: ChunkedMetadataStorageProtocol,
        tokenizer: TransformersTokenizer,
        model: PreTrainedModel,
        dataloader: torch.utils.data.DataLoader,
        embedding_extractor: EmbeddingExtractor,
        single_sequence_batch_size: int = 16,
        device: torch.device = default_device,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Create a Data Handler Class.

        The storage and embedding extraction are handled via dependency injection.
        """
        self.array_storage_backend: ChunkedArrayStorageProtocol = array_storage_backend
        self.metadata_storage_backend: ChunkedMetadataStorageProtocol = metadata_storage_backend

        self.tokenizer: TransformersTokenizer = tokenizer
        self.model: PreTrainedModel = model
        self.dataloader: torch.utils.data.DataLoader = dataloader
        self.embedding_extractor: EmbeddingExtractor = embedding_extractor

        # The batch size used in the MLM masked token mode to iterate over the single repeated sequence.
        self.single_sequence_batch_size: int = single_sequence_batch_size

        self.device: torch.device = device

        self.verbosity: Verbosity = verbosity
        self.logger: logging.Logger = logger

    def process_data(
        self,
    ) -> None:
        """Process the data.

        This worker method opens the storage and iterates over the dataloader.
        """
        self.open_storage()
        self.iterate_over_dataloader()

    def open_storage(
        self,
    ) -> None:
        self.array_storage_backend.open()
        self.metadata_storage_backend.open()

    def iterate_over_dataloader(
        self,
    ) -> None:
        # Iterate over batches and write embeddings to storage
        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg="Computing and storing embeddings ...",
            )

        for batch_idx, batch in enumerate(
            iterable=tqdm(
                self.dataloader,
                desc="Computing and storing embeddings",
            ),
        ):
            self.process_single_batch(
                batch=batch,
                batch_idx=batch_idx,
            )

        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg="Computing and storing embeddings DONE",
            )

    def process_single_batch(
        self,
        batch: dict,
        batch_idx: int,
    ) -> None:
        embeddings: np.ndarray = self.compute_embeddings_from_batch(
            batch=batch,
        )

        chunk_identifier: ChunkIdentifier = self.get_chunk_identifier(
            batch=batch,
            batch_idx=batch_idx,
        )

        # Write embeddings to storage
        array_data_chunk = ArrayDataChunk(
            batch_of_sequences_embedding_array=embeddings,
            chunk_identifier=chunk_identifier,
        )

        self.array_storage_backend.write_chunk(
            data_chunk=array_data_chunk,
        )

        batch_cpu = move_batch_to_cpu(
            batch=batch,
        )

        # Write metadata to storage
        metadata_data_chunk = MetadataChunk(
            batch=batch_cpu,
            chunk_identifier=chunk_identifier,
        )

        self.metadata_storage_backend.write_chunk(
            data_chunk=metadata_data_chunk,
        )

    def get_chunk_identifier(
        self,
        batch: dict,
        batch_idx: int,
    ) -> ChunkIdentifier:
        batch_len: int = self.get_batch_len(
            batch=batch,
        )

        chunk_identifier = ChunkIdentifier(
            chunk_idx=batch_idx,
            start_idx=batch_idx * batch_len,
            chunk_length=batch_len,
        )

        return chunk_identifier

    def get_batch_len(
        self,
        batch: dict,
    ) -> int:
        inputs: dict[
            str,
            torch.Tensor,
        ] = self.prepare_model_inputs_from_batch(
            batch=batch,
        )
        batch_len: int = len(inputs["input_ids"])
        return batch_len

    def prepare_model_inputs_from_batch(
        self,
        batch: dict,
    ) -> dict[
        str,
        torch.Tensor,
    ]:
        """Prepare model inputs from a batch."""
        inputs = batch["model_inputs"]
        return inputs

    @abstractmethod
    def compute_embeddings_from_batch(
        self,
        batch: dict,
    ) -> np.ndarray:
        """Compute model outputs and extract embeddings from a batch."""
        ...  # pragma: no cover

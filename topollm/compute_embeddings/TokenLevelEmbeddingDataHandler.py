# coding=utf-8
#
# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Julius von Rohrscheidt (julius.rohrscheidt@helmholtz-muenchen.de)
# Benjamin Ruppik (ruppik@hhu.de)
#
# This code was generated with the help of AI writing assistants
# including GitHub Copilot, ChatGPT, Bing Chat.
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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Imports

# Standard library imports
import logging

# Third party imports
import numpy as np
import torch
import torch.utils.data
from tqdm.auto import tqdm
from transformers import PreTrainedModel
import transformers.modeling_outputs

# Local imports
from topollm.compute_embeddings.embedding_extractor.EmbeddingExtractorProtocol import (
    EmbeddingExtractor,
)
from topollm.storage.StorageProtocols import (
    ArrayDataChunk,
    ChunkIdentifier,
    ChunkedArrayStorageProtocol,
    ChunkedMetadataStorageProtocol,
    MetaDataChunk,
)

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class TokenLevelEmbeddingDataHandler:
    """
    Create a Data Handler Class with Dependency Injection
    """

    def __init__(
        self,
        array_storage_backend: ChunkedArrayStorageProtocol,
        metadata_storage_backend: ChunkedMetadataStorageProtocol,
        model: PreTrainedModel,
        dataloader: torch.utils.data.DataLoader,
        embedding_extractor: EmbeddingExtractor,
        logger: logging.Logger = logging.getLogger(__name__),
    ):
        self.array_storage_backend = array_storage_backend
        self.metadata_storage_backend = metadata_storage_backend
        self.model = model
        self.dataloader = dataloader
        self.embedding_extractor = embedding_extractor
        self.logger = logger

    def process_data(
        self,
    ) -> None:
        """
        Main method to process the data.
        This method opens the storage and iterates over the dataloader.
        """
        self.open_storage()
        self.iterate_over_dataloader()

        return

    def open_storage(
        self,
    ) -> None:
        self.array_storage_backend.open()
        self.metadata_storage_backend.open()

        return

    def iterate_over_dataloader(
        self,
    ):
        # Iterate over batches and write embeddings to storage
        self.logger.info("Computing and storing embeddings ...")

        for batch_idx, batch in tqdm(
            enumerate(self.dataloader),
            desc="Computing and storing embeddings",
        ):
            self.process_single_batch(
                batch=batch,
                batch_idx=batch_idx,
            )

        self.logger.info("Computing and storing embeddings DONE")

    def process_single_batch(
        self,
        batch: dict,
        batch_idx: int,
    ) -> None:
        embeddings = self.compute_embeddings_from_batch(
            batch=batch,
        )

        chunk_identifier = self.get_chunk_identifier(
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

        # Write metadata to storage
        metadata_data_chunk = MetaDataChunk(
            batch=batch,
            chunk_identifier=chunk_identifier,
        )

        self.metadata_storage_backend.write_chunk(
            data_chunk=metadata_data_chunk,
        )

        return

    def get_chunk_identifier(
        self,
        batch: dict,
        batch_idx: int,
    ) -> ChunkIdentifier:
        batch_len = len(batch["input_ids"])

        chunk_identifier = ChunkIdentifier(
            chunk_idx=batch_idx,
            start_idx=batch_idx * batch_len,
            chunk_size=batch_len,
        )

        return chunk_identifier

    def compute_embeddings_from_batch(
        self,
        batch: dict,
    ) -> np.ndarray:
        """
        Function for computing model outputs and extracting embeddings from a batch.
        """

        model_outputs = self.compute_model_outputs_from_single_inputs(
            inputs=batch,
        )
        embeddings = self.embedding_extractor.extract_embeddings_from_model_outputs(
            model_outputs=model_outputs,
        )

        return embeddings

    def compute_model_outputs_from_single_inputs(
        self,
        inputs: dict,
    ) -> transformers.modeling_outputs.BaseModelOutput:
        """
        Compute embeddings for the given inputs using the given model.
        """

        with torch.no_grad():
            # Compute embeddings.
            # The `output_hidden_states` argument needs to be set to `True`
            # so that we can access the hidden states from the different layers
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
            )

        return outputs

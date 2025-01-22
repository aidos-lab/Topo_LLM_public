# Copyright 2024-2025
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (mail@ruppik.net)
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

"""Worker module to compute embedding vectors and store them to disk."""

import logging
from functools import partial
from typing import TYPE_CHECKING

import torch
import torch.utils.data

from topollm.compute_embeddings.collate_batch_for_embedding import (
    collate_batch_and_move_to_device,
)
from topollm.compute_embeddings.embedding_data_handler.factory import get_embedding_data_handler
from topollm.compute_embeddings.embedding_dataloader_preparer.embedding_dataloader_preparer_context import (
    EmbeddingDataLoaderPreparerContext,
)
from topollm.compute_embeddings.embedding_dataloader_preparer.factory import (
    get_embedding_dataloader_preparer,
)
from topollm.compute_embeddings.embedding_extractor.factory import (
    get_embedding_extractor,
)
from topollm.config_classes.main_config import MainConfig
from topollm.model_handling.prepare_loaded_model_container import (
    prepare_device_and_tokenizer_and_model_from_main_config,
)
from topollm.path_management.embeddings.factory import (
    get_embeddings_path_manager,
)
from topollm.storage.factory import (
    StorageFactory,
    StoragePaths,
    StorageSpecification,
)
from topollm.storage.StorageDataclasses import ArrayProperties
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    from topollm.compute_embeddings.embedding_data_handler.base_embedding_data_handler import BaseEmbeddingDataHandler
    from topollm.compute_embeddings.embedding_dataloader_preparer.protocol import EmbeddingDataLoaderPreparer
    from topollm.compute_embeddings.embedding_extractor.protocol import EmbeddingExtractor
    from topollm.model_handling.loaded_model_container import LoadedModelContainer
    from topollm.path_management.embeddings.protocol import EmbeddingsPathManager
    from topollm.storage.array_storage.protocol import ChunkedArrayStorageProtocol
    from topollm.storage.metadata_storage.protocol import ChunkedMetadataStorageProtocol


default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def compute_and_store_embeddings(
    main_config: MainConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Compute and store embedding vectors."""
    embeddings_path_manager: EmbeddingsPathManager = get_embeddings_path_manager(
        main_config=main_config,
        logger=logger,
    )

    loaded_model_container: LoadedModelContainer = prepare_device_and_tokenizer_and_model_from_main_config(
        main_config=main_config,
        verbosity=verbosity,
        logger=logger,
    )

    # Put the model in evaluation mode.
    # For example, dropout layers behave differently during evaluation.
    loaded_model_container.model.eval()

    # Check that the model config exists as an attribute of the model object.
    if not hasattr(
        loaded_model_container.model,
        "config",
    ):
        msg = (
            "The model object does not have an attribute 'model_config', "
            "which is necessary to access the hidden size of the model."
        )
        raise ValueError(
            msg,
        )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Prepare data collator
    partial_collate_fn = partial(
        collate_batch_and_move_to_device,
        device=loaded_model_container.device,
        model_input_names=loaded_model_container.tokenizer.model_input_names,
    )

    preparer_context = EmbeddingDataLoaderPreparerContext(
        data_config=main_config.data,
        embeddings_config=main_config.embeddings,
        tokenizer_config=main_config.tokenizer,
        tokenizer=loaded_model_container.tokenizer,
        collate_fn=partial_collate_fn,
        verbosity=verbosity,
        logger=logger,
    )
    embedding_dataloader_preparer: EmbeddingDataLoaderPreparer = get_embedding_dataloader_preparer(
        preparer_context=preparer_context,
    )
    dataloader: torch.utils.data.DataLoader = embedding_dataloader_preparer.prepare_dataloader()
    # Note:
    #
    # You can use the following code to access the underlying dataset:
    # `dataset = embedding_dataloader_preparer.dataset_preparer.prepare_dataset()`

    # Number of the sequence of dataset entries
    number_of_sequences: int = len(embedding_dataloader_preparer)
    # Length of each sequence
    length_of_sequence: int = embedding_dataloader_preparer.sequence_length
    # Dimension of the embeddings
    embedding_dimension: int = loaded_model_container.model.config.hidden_size

    array_properties = ArrayProperties(
        shape=(
            number_of_sequences,
            length_of_sequence,
            embedding_dimension,
        ),
        dtype="float32",
        chunks=(main_config.storage.chunk_size,),
    )
    storage_paths = StoragePaths(
        array_dir=embeddings_path_manager.array_dir_absolute_path,
        metadata_dir=embeddings_path_manager.metadata_dir_absolute_path,
    )
    storage_specification = StorageSpecification(
        array_storage_type=main_config.storage.array_storage_type,
        metadata_storage_type=main_config.storage.metadata_storage_type,
        array_properties=array_properties,
        storage_paths=storage_paths,
    )

    storage_factory = StorageFactory(
        storage_specification=storage_specification,
        logger=logger,
    )

    array_storage_backend: ChunkedArrayStorageProtocol = storage_factory.get_array_storage()
    metadata_storage_backend: ChunkedMetadataStorageProtocol = storage_factory.get_metadata_storage()

    embedding_extractor: EmbeddingExtractor = get_embedding_extractor(
        embedding_extraction_config=main_config.embeddings.embedding_extraction,
        model_hidden_size=embedding_dimension,
    )

    embedding_data_handler: BaseEmbeddingDataHandler = get_embedding_data_handler(
        embeddings_config=main_config.embeddings,
        language_model_config=main_config.language_model,
        array_storage_backend=array_storage_backend,
        metadata_storage_backend=metadata_storage_backend,
        tokenizer=loaded_model_container.tokenizer,
        model=loaded_model_container.model,
        dataloader=dataloader,
        embedding_extractor=embedding_extractor,
        device=loaded_model_container.device,
        verbosity=verbosity,
        logger=logger,
    )

    embedding_data_handler.process_data()

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

"""Worker module to compute embedding vectors and store them to disk."""

import logging
from functools import partial

import torch
import torch.utils.data

from topollm.compute_embeddings.collate_batch_for_embedding import (
    collate_batch_and_move_to_device,
)
from topollm.compute_embeddings.embedding_data_handler.TokenLevelEmbeddingDataHandler import (
    TokenLevelEmbeddingDataHandler,
)
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
from topollm.model_handling.model.load_model import load_model
from topollm.model_handling.tokenizer.load_tokenizer import load_modified_tokenizer
from topollm.path_management.embeddings.factory import (
    get_embeddings_path_manager,
)
from topollm.storage.factory import (
    StorageFactory,
    StoragePaths,
    StorageSpecification,
)
from topollm.storage.StorageDataclasses import ArrayProperties

logger = logging.getLogger(__name__)


def compute_and_store_embeddings(
    main_config: MainConfig,
    device: torch.device,
    logger: logging.Logger = logger,
) -> None:
    """Compute and store embedding vectors."""
    tokenizer, tokenizer_modifier = load_modified_tokenizer(
        main_config=main_config,
        logger=logger,
    )

    # Logging of the model happens in the 'load_model' function
    model = load_model(
        pretrained_model_name_or_path=main_config.language_model.pretrained_model_name_or_path,
        device=device,
        logger=logger,
        verbosity=main_config.verbosity,
    )

    # Put the model in evaluation mode.
    # For example, dropout layers behave differently during evaluation.
    model.eval()

    # Potential modification of the tokenizer and the model if this is necessary for compatibility.
    # For instance, for some autoregressive models, the tokenizer needs to be modified to add a padding token.
    model = tokenizer_modifier.update_model(
        model=model,
    )

    # Check that the model config exists as an attribute of the model object.
    if not hasattr(
        model,
        "config",
    ):
        msg = (
            "The model object does not have an attribute 'model_config', "
            "which is necessary to access the hidden size of the model."
        )
        raise ValueError(msg)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Prepare data collator
    partial_collate_fn = partial(
        collate_batch_and_move_to_device,
        device=device,
        model_input_names=tokenizer.model_input_names,
    )

    preparer_context = EmbeddingDataLoaderPreparerContext(
        data_config=main_config.data,
        embeddings_config=main_config.embeddings,
        tokenizer_config=main_config.tokenizer,
        tokenizer=tokenizer,
        collate_fn=partial_collate_fn,
        logger=logger,
        verbosity=main_config.verbosity,
    )
    embedding_dataloader_preparer = get_embedding_dataloader_preparer(
        preparer_context=preparer_context,
    )
    dataloader: torch.utils.data.DataLoader = embedding_dataloader_preparer.prepare_dataloader()

    # Number of the sequence of dataset entries
    number_of_sequences = len(embedding_dataloader_preparer)
    # Length of each sequence
    length_of_sequence = embedding_dataloader_preparer.sequence_length
    # Dimension of the embeddings
    embedding_dimension: int = model.config.hidden_size

    array_properties = ArrayProperties(
        shape=(
            number_of_sequences,
            length_of_sequence,
            embedding_dimension,
        ),
        dtype="float32",
        chunks=(main_config.storage.chunk_size,),
    )

    embeddings_path_manager = get_embeddings_path_manager(
        main_config=main_config,
        logger=logger,
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

    array_storage_backend = storage_factory.get_array_storage()
    metadata_storage_backend = storage_factory.get_metadata_storage()

    embedding_extractor = get_embedding_extractor(
        embedding_extraction_config=main_config.embeddings.embedding_extraction,
        model_hidden_size=embedding_dimension,
    )

    data_handler = TokenLevelEmbeddingDataHandler(
        array_storage_backend=array_storage_backend,
        metadata_storage_backend=metadata_storage_backend,
        model=model,
        dataloader=dataloader,
        embedding_extractor=embedding_extractor,
        logger=logger,
    )
    data_handler.process_data()

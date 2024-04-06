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

import logging
from functools import partial

import torch
import transformers

from topollm.compute_embeddings.collate_batch_for_embedding import (
    collate_batch_and_move_to_device,
)
from topollm.compute_embeddings.embedding_extractor.EmbeddingExtractorFactory import (
    get_embedding_extractor,
)
from topollm.compute_embeddings.embedding_dataloader_preparer.EmbeddingDataLoaderPreparerContext import (
    EmbeddingDataLoaderPreparerContext,
)
from topollm.compute_embeddings.embedding_dataloader_preparer.EmbeddingDataLoaderPreparerFactory import (
    get_embedding_dataloader_preparer,
)
from topollm.compute_embeddings.TokenLevelEmbeddingDataHandler import (
    TokenLevelEmbeddingDataHandler,
)
from topollm.config_classes.MainConfig import MainConfig
from topollm.path_management.embeddings.EmbeddingsPathManagerFactory import (
    get_embeddings_path_manager,
)
from topollm.model_handling.model.load_model import load_model
from topollm.model_handling.tokenizer.load_tokenizer import load_tokenizer
from topollm.storage.StorageDataclasses import ArrayProperties
from topollm.storage.StorageFactory import (
    StorageFactory,
    StoragePaths,
    StorageSpecification,
)


def compute_and_store_embeddings(
    main_config: MainConfig,
    device: torch.device,
    logger: logging.Logger = logging.getLogger(__name__),
) -> None:
    tokenizer = load_tokenizer(
        pretrained_model_name_or_path=main_config.language_model.pretrained_model_name_or_path,
        tokenizer_config=main_config.tokenizer,
        logger=logger,
        verbosity=main_config.verbosity,
    )
    model = load_model(
        pretrained_model_name_or_path=main_config.language_model.pretrained_model_name_or_path,
        device=device,
        logger=logger,
        verbosity=main_config.verbosity,
    )
    # Put the model in evaluation mode.
    # For example, dropout layers behave differently during evaluation.
    model.eval()

    model_config: transformers.PretrainedConfig = model.config
    if model_config is None:
        raise ValueError(
            "Model does not have a configuration",
        )
    if main_config.verbosity >= 1:
        logger.info(
            f"{model_config = }",
        )

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
        dataset_type=main_config.data.dataset_type,
        preparer_context=preparer_context,
    )
    dataloader = embedding_dataloader_preparer.prepare_dataloader()
    # For debugging, you can get the first batch from the dataloader like this:
    # example_batch = next(iter(dataloader))

    # Number of the sequence of dataset entries
    N = len(embedding_dataloader_preparer)
    # Length of each sequence
    S = embedding_dataloader_preparer.sequence_length
    # Dimension of the embeddings
    D: int = model_config.hidden_size

    array_properties = ArrayProperties(
        shape=(N, S, D),
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
        model_hidden_size=D,
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

    return None

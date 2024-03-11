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

"""
Create embedding vectors from dataset.
"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Imports

# Standard library imports
import logging
import os
import pathlib
from functools import partial

# Third party imports
import hydra
import hydra.core.hydra_config
import omegaconf
import torch
import torch.utils.data
import transformers
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

# Local imports
from topollm.compute_embeddings.collate_batch_for_embedding import (
    collate_batch_and_move_to_device,
)
from topollm.compute_embeddings.embedding_extractor.EmbeddingExtractorFactory import (
    get_embedding_extractor,
)
from topollm.compute_embeddings.EmbeddingDataLoaderPreparer import (
    EmbeddingDataLoaderPreparerContext,
    get_embedding_dataloader_preparer,
)
from topollm.compute_embeddings.TokenLevelEmbeddingDataHandler import (
    TokenLevelEmbeddingDataHandler,
)
from topollm.config_classes.Configs import MainConfig
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.storage.StorageFactory import (
    StorageFactory,
    StoragePaths,
    StorageSpecification,
)
from topollm.storage.StorageProtocols import ArrayProperties

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Globals

# A logger for this file
global_logger = logging.getLogger(__name__)

setup_exception_logging(
    logger=global_logger,
)

# END Globals
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def load_tokenizer_and_model(
    pretrained_model_name_or_path: str | os.PathLike,
    device: torch.device,
    logger: logging.Logger = logging.getLogger(__name__),
    verbosity: int = 1,
) -> tuple[PreTrainedTokenizer | PreTrainedTokenizerFast, PreTrainedModel,]:
    """Loads the tokenizer and model based on the configuration,
    and puts the model in evaluation mode.

    Args:
        pretrained_model_name_or_path:
            The name or path of the pretrained model.

    Returns:
        A tuple of (tokenizer, model).
    """
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
    )
    model: PreTrainedModel = AutoModel.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
    )

    model.eval()  # Disable dropout layers
    model.to(device)  # type: ignore

    if verbosity >= 1:
        logger.info(
            f"tokenizer:\n" f"{tokenizer}",
        )
        logger.info(
            f"model:\n" f"{model}",
        )
        logger.info(
            f"{device = }",
        )

    return tokenizer, model


@hydra.main(
    config_path="../../configs",
    config_name="main_config",
    version_base="1.2",
)
def main(
    config: omegaconf.DictConfig,
):
    """Run the script."""

    print("Running script ...")

    global_logger.info("Running script ...")

    main_config: MainConfig = initialize_configuration(
        config=config,
        logger=global_logger,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    compute_embeddings(
        main_config=main_config,
        device=device,
        logger=global_logger,
    )

    global_logger.info("Running script DONE")

    return


def compute_embeddings(
    main_config: MainConfig,
    device: torch.device,
    logger: logging.Logger = logging.getLogger(__name__),
):
    tokenizer, model = load_tokenizer_and_model(
        pretrained_model_name_or_path=main_config.embeddings.huggingface_model_name,
        device=device,
        logger=logger,
        verbosity=main_config.verbosity,
    )
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

    # TODO: Implement these paths
    #
    # storage_paths = StoragePaths(
    #     array_dir=main_config.embeddings.array_dir,
    #     metadata_dir=main_config.embeddings.metadata_dir,
    # )
    storage_paths = StoragePaths(
        array_dir=pathlib.Path(
            "data",
            "embeddings",
            "test_array_dir",
        ),
        metadata_dir=pathlib.Path(
            "data",
            "embeddings",
            "test_metadata_dir",
        ),
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

    return


if __name__ == "__main__":
    main()

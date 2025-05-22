# Copyright 2024-2025
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Matthias Ruppik (mail@ruppik.net)
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
import pprint

import torch
import torch.utils.data
from transformers import PreTrainedModel

from topollm.compute_embeddings.embedding_data_handler.base_embedding_data_handler import BaseEmbeddingDataHandler
from topollm.compute_embeddings.embedding_data_handler.mlm_masked_token_embedding_data_handler import (
    MLMMaskedTokenEmbeddingDataHandler,
)
from topollm.compute_embeddings.embedding_data_handler.regular_token_embedding_data_handler import (
    RegularTokenEmbeddingDataHandler,
)
from topollm.compute_embeddings.embedding_extractor.protocol import EmbeddingExtractor
from topollm.config_classes.embeddings.embeddings_config import EmbeddingDataHandlerConfig, EmbeddingsConfig
from topollm.config_classes.language_model.language_model_config import LanguageModelConfig
from topollm.storage.array_storage.protocol import ChunkedArrayStorageProtocol
from topollm.storage.metadata_storage.protocol import ChunkedMetadataStorageProtocol
from topollm.typing.enums import EmbeddingDataHandlerMode, LMmode, Verbosity
from topollm.typing.types import TransformersTokenizer

default_device: torch.device = torch.device(
    device="cpu",
)
default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def get_embedding_data_handler(
    embeddings_config: EmbeddingsConfig,
    language_model_config: LanguageModelConfig,
    array_storage_backend: ChunkedArrayStorageProtocol,
    metadata_storage_backend: ChunkedMetadataStorageProtocol,
    tokenizer: TransformersTokenizer,
    model: PreTrainedModel,
    dataloader: torch.utils.data.DataLoader,
    embedding_extractor: EmbeddingExtractor,
    device: torch.device = default_device,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> BaseEmbeddingDataHandler:
    """Get an embedding data handler.

    The language model config is used to check that the masking mode is compatible with the model.
    """
    embedding_data_handler_config: EmbeddingDataHandlerConfig = embeddings_config.embedding_data_handler

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "embedding_data_handler_config:\n%s",
            pprint.pformat(embedding_data_handler_config),
        )
        logger.info(
            msg=f"{embedding_data_handler_config.mode = }",  # noqa: G004 - low overhead
        )

    match embedding_data_handler_config.mode:
        case EmbeddingDataHandlerMode.REGULAR:
            embedding_data_handler = RegularTokenEmbeddingDataHandler(
                array_storage_backend=array_storage_backend,
                metadata_storage_backend=metadata_storage_backend,
                tokenizer=tokenizer,
                model=model,
                dataloader=dataloader,
                embedding_extractor=embedding_extractor,
                device=device,
                verbosity=verbosity,
                logger=logger,
            )
        case EmbeddingDataHandlerMode.MASKED_TOKEN:
            if language_model_config.lm_mode != LMmode.MLM:
                msg = "For the MLMMaskedTokenEmbeddingDataHandler, we require a masked language model."
                raise ValueError(
                    msg,
                )

            embedding_data_handler = MLMMaskedTokenEmbeddingDataHandler(
                array_storage_backend=array_storage_backend,
                metadata_storage_backend=metadata_storage_backend,
                tokenizer=tokenizer,
                model=model,
                dataloader=dataloader,
                embedding_extractor=embedding_extractor,
                device=device,
                verbosity=verbosity,
                logger=logger,
            )
        case _:
            msg = f"Unknown {embedding_data_handler_config.mode = }."
            raise ValueError(
                msg,
            )

    return embedding_data_handler

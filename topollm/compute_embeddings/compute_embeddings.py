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

"""
Create embedding vectors.

# TODO This script is under development
"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Imports

# Standard library imports
from cgi import test
from functools import partial
import logging
import os
import pathlib
from dataclasses import dataclass
from os import PathLike
from typing import Protocol

# Third party imports
import hydra
import hydra.core.hydra_config
import numpy as np
import omegaconf
import torch
import torch.utils.data
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.configuration_utils import PretrainedConfig

# Local imports
from topollm.config_classes.Configs import MainConfig, EmbeddingExtractionConfig
from topollm.config_classes.enums import Level, AggregationType
from topollm.storage.TokenLevelEmbeddingStorageFactory import (
    get_token_level_embedding_storage,
)
from topollm.utils.initialize_configuration_and_log import initialize_configuration
from topollm.utils.setup_exception_logging import setup_exception_logging
from topollm.compute_embeddings.EmbeddingDataLoaderPreparer import (
    EmbeddingDataLoaderPreparerContext,
    get_embedding_dataloader_preparer,
)
from topollm.storage.StorageProtocols import (
    ChunkedArrayStorageProtocol,
    ChunkedMetadataStorageProtocol,
    ArrayDataChunk,
    ChunkIdentifier,
    ArrayProperties,
    StoragePaths,
)
from topollm.utils.collate_batch_for_embedding import (
    collate_batch_and_move_to_device,
)

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


def load_tokenizer_and_model_for_embedding(
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


class LayerExtractor(Protocol):
    def extract_layers_from_model_outputs(
        self,
        hidden_states,
    ) -> list[torch.Tensor]:
        """
        This method extracts layers from the model outputs.
        """
        ...


class LayerExtractorFromIndices:
    """
    Implementation of the LayerExtractor protocol
    which is configured from a list of layer indices.
    """

    def __init__(
        self,
        layer_indices: list[int],
    ):
        self.layer_indices = layer_indices

    def extract_layers_from_model_outputs(
        self,
        hidden_states,
    ) -> list[torch.Tensor]:
        layers_to_extract = [hidden_states[i] for i in self.layer_indices]
        return layers_to_extract


class LayerAggregator(Protocol):
    dimension_multiplier: int

    def aggregate_layers(
        self,
        layers_to_extract: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        This method aggregates the layers to be extracted into a single tensor.
        """
        ...


class MeanLayerAggregator:
    """
    Implementation of the LayerAggregator protocol
    which computes the mean of the layers to be extracted.
    """

    def __init__(self):
        self.dimension_multiplier = 1

    def aggregate_layers(
        self,
        layers_to_extract: list[torch.Tensor],
    ) -> torch.Tensor:
        # Mean across the layers
        aggregated_layers = torch.mean(
            torch.stack(
                layers_to_extract,
                dim=0,
            ),
            dim=0,
        )
        return aggregated_layers


class ConcatenateLayerAggregator:
    """
    Implementation of the LayerAggregator protocol
    which concatenates the layers to be extracted.
    """

    def __init__(self):
        self.dimension_multiplier = (
            1  # ! TODO: This needs to be set flexibly depending on the method
        )

    def aggregate_layers(
        self,
        layers_to_extract: list[torch.Tensor],
    ) -> torch.Tensor:
        # Concatenate across the last dimension
        aggregated_layers = torch.cat(
            layers_to_extract,
            dim=-1,
        )
        return aggregated_layers


class EmbeddingExtractor(Protocol):
    def extract_embeddings_from_model_outputs(
        self,
        model_outputs,
    ) -> np.ndarray:
        """
        This method extracts embeddings from the model outputs.
        """
        ...

    def embedding_dimension(
        self,
        model_hidden_dimension: int,
    ) -> int:
        """
        Given a model hidden dimension, this method returns the dimension of the embeddings
        which will be computed by the model combined with the extraction method.
        """
        ...


class TokenLevelEmbeddingExtractor:
    """
    Implementation of the EmbeddingExtractor protocol
    which extracts token level embeddings.
    """

    def __init__(
        self,
        layer_extractor: LayerExtractor,
        layer_aggregator: LayerAggregator,
    ):
        self.layer_extractor = layer_extractor
        self.layer_aggregator = layer_aggregator

    def extract_embeddings_from_model_outputs(
        self,
        model_outputs,
    ) -> np.ndarray:
        # Ensure the model outputs hidden states
        if not hasattr(
            model_outputs,
            "hidden_states",
        ):
            raise ValueError("Model outputs do not contain 'hidden_states'")

        hidden_states = (
            model_outputs.hidden_states
        )  # Assuming this is a tuple of tensors

        # Extract specified layers
        layers_to_extract = self.layer_extractor.extract_layers_from_model_outputs(
            hidden_states=hidden_states,
        )

        # Aggregate the extracted layers
        embeddings = self.layer_aggregator.aggregate_layers(
            layers_to_extract=layers_to_extract,
        )

        return embeddings.cpu().numpy()

    def embedding_dimension(
        self,
        model_hidden_dimension: int,
    ) -> int:
        # TODO: Solve the problem that we somewhere need to determine the dimension of the extracted embeddings

        result = model_hidden_dimension * self.layer_aggregator.dimension_multiplier

        return result


def get_embedding_extractor(
    embedding_extraction_config: EmbeddingExtractionConfig,
    model_config: PretrainedConfig,
) -> EmbeddingExtractor:
    layer_extractor = LayerExtractorFromIndices(
        layer_indices=embedding_extraction_config.layer_indices,
    )

    if embedding_extraction_config.aggregation == AggregationType.MEAN:
        layer_aggregator = MeanLayerAggregator()
    elif embedding_extraction_config.aggregation == AggregationType.CONCATENATE:
        layer_aggregator = ConcatenateLayerAggregator()
    else:
        raise ValueError(
            f"Unknown aggregation method: "
            f"{embedding_extraction_config.aggregation = }",
        )

    embedding_extractor = TokenLevelEmbeddingExtractor(
        layer_extractor=layer_extractor,
        layer_aggregator=layer_aggregator,
    )

    return embedding_extractor


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
        self.storage = array_storage_backend
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
        self.storage.open()

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

        array_data_chunk = ArrayDataChunk(
            batch_of_sequences_embedding_array=embeddings,
            chunk_identifier=chunk_identifier,
        )

        # TODO Save metadata (i.e., the batch)

        self.storage.write_chunk(
            data_chunk=array_data_chunk,
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
        )

        return chunk_identifier

    def compute_embeddings_from_batch(
        self,
        batch: dict,
    ) -> np.ndarray:
        # Adjusted function for computing embeddings that directly writes to array

        # Compute embeddings
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
    ):
        """
        Compute embeddings for the given inputs using the given model.
        """

        # Compute embeddings
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
            )

        return outputs


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
    tokenizer, model = load_tokenizer_and_model_for_embedding(
        pretrained_model_name_or_path=main_config.embeddings.huggingface_model_name,
        device=device,
        logger=logger,
        verbosity=main_config.verbosity,
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
    D = model.config.hidden_size

    array_properties = ArrayProperties(
        shape=(N, S, D),
        dtype="float32",
        chunks=(1024,),  # TODO: Make chunk size configurable
    )

    # TODO: Implement these paths
    #
    # storage_paths = StoragePaths(
    #     array_dir=main_config.embeddings.array_dir,
    #     metadata_dir=main_config.embeddings.metadata_dir,
    # )
    storage_paths = StoragePaths(
        array_dir=pathlib.Path("test_array_dir"),
        metadata_dir=pathlib.Path("test_metadata_dir"),
    )

    storage_backend = get_token_level_embedding_storage(
        storage_type=main_config.storage.storage_type,
        array_properties=array_properties,
        storage_paths=storage_paths,
    )

    embedding_extractor = get_embedding_extractor(
        embedding_extraction_config=main_config.embeddings.embedding_extraction,
        model_config=model.config,
    )

    data_handler = TokenLevelEmbeddingDataHandler(
        array_storage_backend=storage_backend,
        model=model,
        dataloader=dataloader,
        embedding_extractor=embedding_extractor,
        logger=logger,
    )
    data_handler.process_data()

    return


if __name__ == "__main__":
    main()

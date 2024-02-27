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
from functools import partial
import logging
import os
from dataclasses import dataclass
from os import PathLike
from typing import Protocol, runtime_checkable

# Third party imports
import hydra
import hydra.core.hydra_config
import numpy as np
import omegaconf
import torch
import torch.utils.data
import zarr
import zarr.core
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from topollm.compute_embeddings.EmbeddingDataLoaderPreparer import (
    EmbeddingDataLoaderPreparerContext,
    get_embedding_dataloader_preparer,
)


# Local imports
from topollm.config_classes.Configs import MainConfig
from topollm.config_classes.enums import Level
from topollm.utils.initialize_configuration_and_log import initialize_configuration
from topollm.utils.setup_exception_logging import setup_exception_logging

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


def collate_batch(
    batch: list,
) -> dict:
    """
    Function to collate the batch.

    Args:
        batch (list):
    Returns:
        features (dict): Features for the batch.
    """
    collated_batch: dict[str, torch.Tensor] = {
        "input_ids": torch.tensor([item["input_ids"] for item in batch]),
        "attention_mask": torch.tensor([item["attention_mask"] for item in batch]),
    }

    return collated_batch


def move_collated_batch_to_device(
    collated_batch: dict,
    device: torch.device,
    model_input_names: list[str],
):
    collated_batch = {
        key: value.to(device)
        for key, value in collated_batch.items()
        if key in model_input_names
    }

    return collated_batch


def collate_batch_and_move_to_device(
    batch: list,
    device: torch.device,
    model_input_names: list[str],
) -> dict:
    collated_batch = collate_batch(
        batch=batch,
    )
    collated_batch = move_collated_batch_to_device(
        collated_batch=collated_batch,
        device=device,
        model_input_names=model_input_names,
    )

    return collated_batch


def load_tokenizer_and_model_for_embedding(
    pretrained_model_name_or_path: str | os.PathLike,
    device: torch.device,
    logger: logging.Logger = logging.getLogger(__name__),
    verbosity: int = 1,
) -> tuple[
    PreTrainedTokenizer | PreTrainedTokenizerFast,
    PreTrainedModel,
    torch.device,
]:
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

    return tokenizer, model, device


@dataclass
class DataChunk:
    """
    Dataclass to hold one embedding chunk
    and the batch containing the corresponding dataset entries.
    """

    batch_of_sequences_embedding_array: np.ndarray
    batch: dict
    chunk_idx: int
    start_idx: int


@dataclass
class ArrayProperties:
    shape: tuple[int, int]
    dtype: str
    chunks: tuple[int, int]


# Define a Storage Protocol
@runtime_checkable
class EmbeddingStorageProtocol(Protocol):
    def open(
        self,
        array_properties: ArrayProperties,
    ) -> None:
        """Initializes the storage with specified configuration."""
        ...

    def write_chunk(
        self,
        data_chunk: DataChunk,
    ) -> None:
        """Writes a chunk of data starting from a specific index."""
        ...

    def read_chunk(
        self,
        start_idx: int,
        end_idx: int,
    ) -> DataChunk:
        """Reads a chunk of data starting from a specific index."""
        ...


def get_embedding_storage(
    storage_type: str,
    store_dir: PathLike,
) -> EmbeddingStorageProtocol:
    """Factory function to instantiate storage backends based on the storage type.

    Args:
        storage_type:
            The type of storage to use.
        store_dir:
            The directory to store the embeddings in.

    Returns:
        An instance of a storage backend.
    """
    if storage_type == "zarr":
        return ZarrEmbeddingStorage(
            store_dir=store_dir,
        )
    # Extendable to other storage types
    # elif storage_type == "hdf5":
    #     return Hdf5EmbeddingStorage(store_dir)
    else:
        raise ValueError(f"Unsupported {storage_type = }")


# Implement the Protocol with a Zarr Storage Class
class ZarrEmbeddingStorage:
    def __init__(
        self,
        store_dir: PathLike,
    ):
        self.store_dir = store_dir
        self.zarr_array = None

    def open(
        self,
        shape: tuple[int, int],
        dtype: str,
        chunks: tuple[int, int],
    ) -> None:
        os.makedirs(
            self.store_dir,
            exist_ok=True,
        )
        self.zarr_array = zarr.open(
            store=self.store_dir,  # type: ignore
            mode="w",
            shape=shape,
            dtype=dtype,
            chunks=chunks,
        )

    def write_chunk(
        self,
        data_chunk: DataChunk,
    ) -> None:
        # TODO: Update this to work with the DataClass

        self.zarr_array[start_idx : start_idx + len(data)] = data


class TokenLevelEmbeddingDataHandler:
    """
    Create a Data Handler Class with Dependency Injection
    """

    # TODO: Update this

    def __init__(
        self,
        storage_backend: EmbeddingStorageProtocol,
    ):
        self.storage = storage_backend

    def process_data(
        self,
    ) -> None:
        self.open_storage()
        self.iterate_over_dataloader()

        return

    def open_storage(
        self,
        array_properties: ArrayProperties,
    ) -> None:
        N = len(dataloader.dataset)
        D = 768  # Dimensionality should ideally be determined dynamically

        # self.storage.open(
        #     shape=(N, D),
        #     dtype="float32",
        #     chunks=(1024, D),
        # )

        self.storage.open(
            array_properties,
        )

        return

    def iterate_over_dataloader(
        self,
        dataloader,
        model,
        config,
    ):
        # Iterate over batches and write embeddings to storage
        global_logger.info("Computing and storing embeddings ...")

        start_idx = 0
        for batch in tqdm(
            dataloader,
            desc="Computing and storing embeddings",
        ):
            self.process_single_batch(
                model,
                start_idx,
                batch,
            )
            start_idx += len(batch)

        global_logger.info("Computing and storing embeddings DONE")

    def process_single_batch(
        self,
        model,
        start_idx,
        batch,
    ):
        embeddings = self.compute_embeddings_from_batch(
            batch=batch,
            model=model,
            level=embeddings_config.level,
        )
        self.storage.write_chunk(
            embeddings,
            start_idx,
        )

    def compute_embeddings_from_batch(
        batch: dict,
        model: PreTrainedModel,
        level: Level,
        start_idx: int,
    ):
        # Adjusted function for computing embeddings that directly writes to array

        # Compute embeddings
        outputs = compute_embeddings_from_single_inputs(
            inputs=batch,
            model=model,
            level=level,
        )
        # TODO Extract the correct layer here/potentially aggregate

        return embeddings

    def compute_model_outputs_from_single_inputs(
        inputs: dict,
        model: PreTrainedModel,
        level: Level,
    ) -> np.ndarray:
        """
        Compute embeddings for the given inputs using the given model.
        """

        # Compute embeddings
        with torch.no_grad():
            outputs = model(**inputs)

        return outputs

    def extract_embeddings_from_model_outputs(
        outputs,
    ) -> np.ndarray:
        # Return embeddings
        # TODO: Include the correct layer here/define config for layer extraction
        # TODO: Remove aggregation from here, we will do this in a separate class
        if level == Level.TOKEN:
            return outputs.last_hidden_state.cpu().numpy()
        elif level == Level.DATASET_ENTRY:
            # TODO: Include other aggregation methods here
            return outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        else:
            raise ValueError(f"Unknown {level = }")


@hydra.main(
    config_path="../../configs",
    config_name="main_config",
    version_base="1.2",
)
def main(
    config: omegaconf.DictConfig,
):
    """Run the script."""

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
    tokenizer, model, device = load_tokenizer_and_model_for_embedding(
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

    # TODO: Continue here

    # TODO: Create storage backend
    # TODO: Create data handler and call


if __name__ == "__main__":
    main()

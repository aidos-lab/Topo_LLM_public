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
import logging
import os
import sys
from abc import ABC, abstractmethod
from functools import partial

# Third party imports
import datasets
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
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from topollm.config_classes.Configs import DataConfig, EmbeddingsConfig

# Local imports
from topollm.config_classes.enums import Level, DatasetType
from topollm.utils.setup_main_config_and_log import setup_main_config_and_log

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Globals

os.environ["HYDRA_FULL_ERROR"] = "1"

# A logger for this file
global_logger = logging.getLogger(__name__)


def handle_exception(
    exc_type,
    exc_value,
    exc_traceback,
):
    if issubclass(
        exc_type,
        KeyboardInterrupt,
    ):
        sys.__excepthook__(
            exc_type,
            exc_value,
            exc_traceback,
        )
        return
    else:
        # Log the exception
        global_logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )


sys.excepthook = handle_exception


# END Globals
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def convert_dataset_entry_to_features(
    dataset_entry: dict,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    column_name: str = "text",
    max_length: int = 512,
) -> BatchEncoding:
    """
    Convert dataset entires/examples to features
    by tokenizing the text and padding/truncating to a maximum length.
    """

    features = tokenizer(
        dataset_entry[column_name],
        max_length=max_length,
        padding="max_length",
        truncation="longest_first",
    )

    return features


def collate_batch_and_move_to_device(
    batch: list,
    device: torch.device,
    model_input_names: list[str],
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

    # Move batch to device
    inputs = {
        key: value.to(device)
        for key, value in collated_batch.items()
        if key in model_input_names
    }

    return inputs


# Function to compute embeddings
def compute_embeddings(
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

    # Return embeddings
    # TODO: Include the correct layer here
    if level == Level.TOKEN:
        return outputs.last_hidden_state.cpu().numpy()
    elif level == Level.DATASET_ENTRY:
        # TODO: Include other aggregation methods here
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    else:
        raise ValueError(f"Unknown {level = }")


def process_embedding_batch(
    batch: dict,
    model: PreTrainedModel,
    level: Level,
    zarr_array: zarr.core.Array,
    start_idx: int,
):
    # Adjusted function for computing embeddings that directly writes to array

    # Compute embeddings
    embeddings = compute_embeddings(
        inputs=batch,
        model=model,
        level=level,
    )

    # TODO Extract the correct layer here/potentially aggregate

    # TODO Write embeddings and metadata to disk

    # Write embeddings to the array
    zarr_array[
        start_idx : start_idx + embeddings.shape[0],
        :,
    ] = embeddings


def load_tokenizer_and_model_for_embedding(
    pretrained_model_name_or_path: str | os.PathLike,
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


class EmbeddingDataLoaderPreparer(ABC):
    """Abstract base class for embedding dataset preparers."""

    def __init__(
        self,
        data_config: DataConfig,
        embeddings_config: EmbeddingsConfig,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        device: torch.device,
        collate_fn,
    ):
        self.data_config = data_config
        self.embeddings_config = embeddings_config
        self.tokenizer = tokenizer
        self.device = device
        self.collate_fn = collate_fn

    @abstractmethod
    def load_and_prepare_dataset(
        self,
    ):
        """Loads and prepares a dataset."""
        pass


class HuggingfaceEmbeddingDataLoaderPreparer(EmbeddingDataLoaderPreparer):
    def load_dataset_dict(
        self,
    ) -> datasets.DatasetDict:
        """Loads the dataset based from huggingface datasets based on configuration."""
        dataset_dict = datasets.load_dataset(
            self.data_config.dataset_identifier,
            trust_remote_code=True,
        )

        if not isinstance(
            dataset_dict,
            datasets.DatasetDict,
        ):
            raise ValueError(
                f"Expected {dataset_dict = } " f"to be a {datasets.DatasetDict = }"
            )

        return dataset_dict

    def select_dataset(
        self,
        dataset_dict: datasets.DatasetDict,
    ) -> datasets.Dataset:
        # Select the dataset split to use
        dataset: datasets.Dataset = dataset_dict[self.data_config.split]

        # Truncate the dataset to the specified number of samples
        dataset = dataset.select(
            indices=range(self.data_config.number_of_samples),
        )

        return dataset

    def prepare_dataset_tokenized(
        self,
        dataset,
    ) -> datasets.Dataset:
        """Tokenizes dataset."""
        # Make a partial function for mapping tokenizer over the dataset
        partial_map_fn = partial(
            convert_dataset_entry_to_features,
            tokenizer=self.tokenizer,
            column_name=self.data_config.column_name,
            max_length=self.embeddings_config.max_length,
        )

        dataset_tokenized = dataset.map(
            partial_map_fn,
            batched=True,
            batch_size=self.embeddings_config.dataset_map.batch_size,
            num_proc=self.embeddings_config.dataset_map.num_proc,
        )

        return dataset_tokenized

    def prepare_dataloader(
        self,
        dataset_tokenized: datasets.Dataset,
    ) -> torch.utils.data.DataLoader:
        # The mapped dataset has the input_ids and attention_mask
        # as lists of integers, but we want to convert them to torch tensors
        # to use them as model input.
        # We will take care of this in the collate function of the DataLoader,
        # which will also move the data to the appropriate device.
        #
        # An alternative way to set the format of the dataset to torch tensors
        # is given below:
        #
        # dataset_tokenized.set_format(
        #     type="torch",
        #     columns=[
        #         "input_ids",
        #         "attention_mask",
        #     ],
        # )

        partial_collate_fn = partial(
            self.collate_fn,
            device=self.device,
            model_input_names=self.tokenizer.model_input_names,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset_tokenized,  # type: ignore
            batch_size=self.embeddings_config.batch_size,
            shuffle=False,
            collate_fn=partial_collate_fn,
        )

        return dataloader


def get_embedding_dataloader_preparer(
    dataset_type: DatasetType,
    data_config: DataConfig,
    embeddings_config: EmbeddingsConfig,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    device: torch.device,
    collate_fn,
) -> EmbeddingDataLoaderPreparer:
    """Factory function to instantiate dataset preparers based on the dataset type.

    Args:
        dataset_type:
            The type of dataset to prepare.
        config:
            Configuration object containing dataset and model settings.
        tokenizer:
            Tokenizer object for datasets that require tokenization.

    Returns:
        An instance of a DatasetPreparer subclass.
    """
    if dataset_type == DatasetType.HUGGINGFACE_DATASET:
        return HuggingfaceEmbeddingDataLoaderPreparer(
            data_config=data_config,
            embeddings_config=embeddings_config,
            tokenizer=tokenizer,
            device=device,
            collate_fn=collate_fn,
        )
    # Extendable to other dataset types
    # elif dataset_type == "convlab_unified_format":
    #     return ImageDatasetPreparer(config)
    else:
        raise ValueError(f"Unsupported {dataset_type = }")


@hydra.main(
    config_path="../../configs",
    config_name="main_config",
    version_base="1.2",
)
def main(
    config: omegaconf.DictConfig,
):
    """Run the script."""

    main_config = setup_main_config_and_log(
        config=config,
        logger=global_logger,
    )

    tokenizer, model, device = load_tokenizer_and_model_for_embedding(
        pretrained_model_name_or_path=main_config.embeddings.huggingface_model_name,
        logger=global_logger,
        verbosity=main_config.verbosity,
    )

    # TODO: Continue here

    N = len(dataset_tokenized["train"])  # Total number of items in the dataset
    D = 768  # Dimensionality of RoBERTa embeddings (for 'roberta-base') # TODO: Change

    # Create a directory for the Zarr store, if it doesn't already exist
    # TODO: Change this
    zarr_dir = "embeddings.zarr"
    os.makedirs(
        zarr_dir,
        exist_ok=True,
    )

    # Initialize a Zarr array
    zarr_array = zarr.open(
        store=zarr_dir,
        mode="w",
        shape=(N, D),
        dtype=np.float32,
        chunks=(1024, D),
    )

    # Iterate over batches and write embeddings
    global_logger.info("Computing and storing embeddings ...")

    start_idx = 0
    for batch in tqdm(
        embedding_dataloader,
        desc="Computing and storing embeddings",
    ):
        process_embedding_batch(
            batch=batch,
            model=model,
            level=main_config.embeddings.level,
            zarr_array=zarr_array,
            start_idx=start_idx,
        )
        start_idx += batch_size

    global_logger.info("Computing and storing embeddings DONE")

    global_logger.info("Script finished.")

    return


if __name__ == "__main__":
    main()

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
import argparse
import logging
import numpy as np
import os
import pathlib
import pprint
from functools import partial

# Third party imports
import datasets
import hydra
import hydra.core.hydra_config
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
    BatchEncoding,
)

# Local imports
from topollm.config_classes.Configs import DataConfig, EmbeddingsConfig, MainConfig
from topollm.config_classes.enums import Level

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Globals

# A logger for this file
global_logger = logging.getLogger(__name__)

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


def collate(
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
    features: dict[str, torch.Tensor] = {
        "input_ids": torch.tensor([item["input_ids"] for item in batch]),
        "attention_mask": torch.tensor([item["attention_mask"] for item in batch]),
    }

    # Move batch to device
    inputs = {
        key: value.to(device)
        for key, value in features.items()
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


@hydra.main(
    config_path="../../configs",
    config_name="main_config",
    version_base="1.2",
)
def main(
    config,
):
    """Run the script."""
    verbosity: int = config.verbosity

    if verbosity >= 1:
        global_logger.info(f"Working directory:\n" f"{os.getcwd() = }")
        global_logger.info(
            f"Hydra output directory:\n"
            f"{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}"
        )
        global_logger.info(
            f"hydra config:\n" f"{pprint.pformat(config)}",
        )

    main_config = MainConfig.model_validate(
        config,
    )

    if verbosity >= 1:
        global_logger.info(
            f"master_config:\n" f"{pprint.pformat(main_config)}",
        )

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=main_config.embeddings.huggingface_model_name,
    )
    model: PreTrainedModel = AutoModel.from_pretrained(
        pretrained_model_name_or_path=main_config.embeddings.huggingface_model_name,
    )

    if verbosity >= 1:
        global_logger.info(
            f"tokenizer:\n" f"{tokenizer}",
        )
        global_logger.info(
            f"model:\n" f"{model}",
        )

    # Load the dataset from huggingface datasets
    dataset = datasets.load_dataset(
        main_config.data.dataset_identifier,
        trust_remote_code=True,
    )

    # TODO: Create split here
    # split=data_config.split,

    # Tokenize the dataset
    partial_convert_dataset_entry_to_features = partial(
        convert_dataset_entry_to_features,
        tokenizer=tokenizer,
        column_name=main_config.data.column_name,
        max_length=main_config.embeddings.max_length,
    )

    dataset_tokenized = dataset.map(
        partial_convert_dataset_entry_to_features,
        batched=True,
        batch_size=main_config.embeddings.dataset_map.batch_size,
        num_proc=2,  # type: ignore
    )

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

    # Initialize the DataLoader
    batch_size = main_config.embeddings.batch_size

    embedding_dataloader = torch.utils.data.DataLoader(
        dataset_tokenized,
        batch_size=batch_size,
        shuffle=False,
    )

    # Ensure the model is in evaluation mode, which disables dropout layers
    model.eval()

    # Move model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_logger.info(f"{device = }")
    model.to(device)  # type: ignore

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

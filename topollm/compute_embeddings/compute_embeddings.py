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
import tqdm
import zarr
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

# Local imports
from topollm.config_classes.Configs import DataConfig, EmbeddingsConfig

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Globals

# A logger for this file
global_logger = logging.getLogger(__name__)

# END Globals
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def tokenize(
    dataset_entry,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    column_name="text",
):
    return tokenizer(
        dataset_entry[column_name],
    )


# Function to compute embeddings
def compute_embeddings(batch):
    """
    # TODO Update this
    """

    # Move batch to device
    inputs = {
        k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names
    }

    # Compute embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # Return embeddings
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()


# Adjusted function for computing embeddings that directly writes to Zarr
def compute_and_store_embeddings(
    batch,
    zarr_array,
    start_idx,
):
    # Compute embeddings as before
    embeddings = compute_embeddings(
        batch
    )  # Assuming this function is defined as before

    # Write embeddings to the Zarr array
    zarr_array[
        start_idx : start_idx + embeddings.shape[0],
        :,
    ] = embeddings


@hydra.main(
    config_path="../../configs",
    config_name="config",
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

    embeddings_config = EmbeddingsConfig.model_validate(
        config.embeddings,
    )
    data_config = DataConfig.model_validate(
        config.data,
    )

    if verbosity >= 1:
        global_logger.info(
            f"embeddings_config:\n" f"{pprint.pformat(embeddings_config)}",
        )
        global_logger.info(
            f"data_config:\n" f"{pprint.pformat(data_config)}",
        )

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=embeddings_config.huggingface_model_name,
    )
    model: PreTrainedModel = AutoModel.from_pretrained(
        pretrained_model_name_or_path=embeddings_config.huggingface_model_name,
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
        data_config.dataset_identifier,
        trust_remote_code=True,
    )

    # TODO: Create split here
    # split=data_config.split,

    # Tokenize the dataset
    partial_tokenize = partial(
        tokenize,
        tokenizer=tokenizer,
        column_name=data_config.column_name,
    )

    dataset_tokenized = dataset.map(
        partial_tokenize,
        batched=True,
    )

    dataset_tokenized.set_format(
        type="torch",
        columns=[
            "input_ids",
            "token_type_ids",
            "attention_mask",
            "label",
        ],
    )
    global_logger.info(f"{dataset_tokenized.format['type'] = }")

    # Initialize the DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset_tokenized,
        batch_size=batch_size,
        shuffle=False,
    )

    # Ensure the model is in evaluation mode, which disables dropout layers
    model.eval()

    # Move model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_logger.info(f"{device = }")
    model.to(device)

    N = len(tokenized_datasets["train"])  # Total number of items in the dataset
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
    start_idx = 0
    for batch in tqdm(train_dataloader):
        compute_and_store_embeddings(batch, start_idx)
        start_idx += batch_size

    return


if __name__ == "__main__":
    main()

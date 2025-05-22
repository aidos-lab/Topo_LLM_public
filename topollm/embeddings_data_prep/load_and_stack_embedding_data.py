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
import pathlib

import numpy as np
import pandas as pd
import zarr

from topollm.config_classes.data_processing_column_names.data_processing_column_names import DataProcessingColumnNames
from topollm.embeddings_data_prep.load_pickle_files_from_meta_path import load_pickle_files_from_meta_path
from topollm.path_management.embeddings.protocol import EmbeddingsPathManager
from topollm.typing.enums import Verbosity

default_logger = logging.getLogger(__name__)
default_data_processing_column_names = DataProcessingColumnNames()


def load_and_stack_embedding_data(
    embeddings_path_manager: EmbeddingsPathManager,
    data_processing_column_names: DataProcessingColumnNames = default_data_processing_column_names,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> pd.DataFrame:
    """Load the embedding data and metadata."""
    # Path for loading the precomputed embeddings
    array_path = embeddings_path_manager.array_dir_absolute_path

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "array_path:%s",
            array_path,
        )

    if not array_path.exists():
        msg = f"{array_path = } does not exist."
        raise FileNotFoundError(
            msg,
        )

    # Path for loading the precomputed metadata
    meta_path = pathlib.Path(
        embeddings_path_manager.metadata_dir_absolute_path,
        "pickle_chunked_metadata_storage",
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "meta_path:%s",
            meta_path,
        )

    if not meta_path.exists():
        msg = f"{meta_path = } does not exist."
        raise FileNotFoundError(
            msg,
        )

    array_zarr = zarr.open(
        store=str(array_path),
        mode="r",
    )

    array_np = np.array(
        array_zarr,
    )
    number_of_sentences = array_np.shape[0]
    number_of_tokens_per_sentence = array_np.shape[1]

    array_np = array_np.reshape(
        array_np.shape[0] * array_np.shape[1],
        array_np.shape[2],
    )

    loaded_metadata = load_pickle_files_from_meta_path(
        meta_path=meta_path,
    )

    if verbosity >= Verbosity.DEBUG:
        logger.info(
            "Loaded pickle files loaded_data:\n%s",
            loaded_metadata,
        )

    # Note: This assumes that the batches saved in the embedding data computation are a dict which contains
    # different keys for the model_inputs and the metadata.
    input_ids_collection: list[list] = [
        metadata_chunk["model_inputs"]["input_ids"].tolist() for metadata_chunk in loaded_metadata
    ]

    stacked_input_ids: np.ndarray = np.vstack(
        input_ids_collection,
    )
    stacked_input_ids: np.ndarray = stacked_input_ids.reshape(
        stacked_input_ids.shape[0] * stacked_input_ids.shape[1],
    )

    sentence_idx = np.array(
        [np.ones(number_of_tokens_per_sentence) * i for i in range(number_of_sentences)],
    ).reshape(
        number_of_sentences * number_of_tokens_per_sentence,
    )

    # Note:
    # The "metadata" key in the batch saved in a metadata_chunk is currently lost.
    # You would need to add it to the full_df here if you want to use it in the downstream pipeline.

    full_data_dict = {
        data_processing_column_names.embedding_vectors: list(array_np),
        data_processing_column_names.token_id: list(stacked_input_ids),
        data_processing_column_names.sentence_idx: [int(x) for x in sentence_idx],
    }

    # # # #
    # Manually concatenate the elements in the POS metadata (if it exist)
    if "POS" in loaded_metadata[0]["metadata"]:
        # POS collection is a
        # list of batches, where each batch is a list of sentences, where each sentence is a list of POS tags
        pos_collection: list[list[list]] = [metadata_chunk["metadata"]["POS"] for metadata_chunk in loaded_metadata]

        concatenated_pos_batches: list = [pos_batch for pos_batches in pos_collection for pos_batch in pos_batches]

        # Concatenate the list of POS tags
        concatenated_pos_tags: list = [pos_tag for pos_tags in concatenated_pos_batches for pos_tag in pos_tags]

        full_data_dict["POS"] = concatenated_pos_tags

    full_df = pd.DataFrame(
        data=full_data_dict,
    )

    return full_df

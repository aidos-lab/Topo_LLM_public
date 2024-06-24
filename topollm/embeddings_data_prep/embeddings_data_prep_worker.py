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

"""Prepare the embedding data of a model and its metadata for further analysis."""

import logging
import pathlib
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import zarr

from topollm.config_classes.main_config import MainConfig
from topollm.embeddings_data_prep.load_pickle_files_from_meta_path import load_pickle_files_from_meta_path
from topollm.logging.log_dataframe_info import log_dataframe_info
from topollm.model_handling.tokenizer.load_tokenizer import load_modified_tokenizer
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.path_management.embeddings.protocol import EmbeddingsPathManager
from topollm.typing.enums import Verbosity
from topollm.typing.types import TransformersTokenizer

default_logger = logging.getLogger(__name__)


def embeddings_data_prep_worker(
    main_config: MainConfig,
    device: torch.device,  # noqa: ARG001 - placeholder for future use
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Prepare the embedding data of a model and its metadata for further analysis."""
    embeddings_path_manager: EmbeddingsPathManager = get_embeddings_path_manager(
        main_config=main_config,
        logger=logger,
    )

    full_df = load_embedding_data(
        embeddings_path_manager=embeddings_path_manager,
        verbosity=verbosity,
        logger=logger,
    )

    tokenizer, arr_no_pad, meta_no_pad, sentence_idx_no_pad = remove_padding_and_extra_tokens(
        full_df=full_df,
        main_config=main_config,
        verbosity=verbosity,
        logger=logger,
    )

    # Choose size of a meta sample which is used to take subsets for a point-wise
    # comparison of local estimators.
    # Sample size of the arrays
    sample_size = main_config.embeddings_data_prep.num_samples

    arr_no_pad, meta_no_pad, sentence_idx_no_pad = select_subsets_of_arr_and_meta(
        arr_no_pad=arr_no_pad,
        meta_no_pad=meta_no_pad,
        sentence_idx_no_pad=sentence_idx_no_pad,
        sample_size=sample_size,
        verbosity=verbosity,
        logger=logger,
    )

    # # # #
    # Convert the metadata to a DataFrame

    # x of type 'numpy.int64' needs to be explicitly converted to an integer,
    # otherwise the convert_ids_to_tokens() method will raise the error:
    # TypeError: 'numpy.int64' object is not iterable
    meta_names = [tokenizer.convert_ids_to_tokens(int(x)) for x in list(full_df.meta)]
    full_df["meta_name"] = meta_names

    token_names_no_pad = [tokenizer.convert_ids_to_tokens(int(x)) for x in meta_no_pad]

    meta_frame = pd.DataFrame(
        {
            "token_id": list(meta_no_pad),
            "token_name": list(token_names_no_pad),
            "sentence_idx": list(sentence_idx_no_pad),
        },
    )

    grouped_df = full_df.iloc[:,1:].groupby('sentence_idx')['meta_name'].apply(' '.join).reset_index()
    meta_frame = pd.merge(meta_frame, grouped_df, on='sentence_idx')

    if verbosity >= Verbosity.NORMAL:
        log_dataframe_info(
            meta_frame,
            df_name="meta_frame",
            logger=logger,
        )

    # # # #
    # Save the prepared data
    save_prepared_data(
        embeddings_path_manager=embeddings_path_manager,
        arr_no_pad=arr_no_pad,
        meta_frame=meta_frame,
        verbosity=verbosity,
        logger=logger,
    )


def load_embedding_data(
    embeddings_path_manager: EmbeddingsPathManager,
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
    sentence_num = array_np.shape[0]
    token_num = array_np.shape[1]

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

    input_ids_collection: list[list] = [metadata_chunk["input_ids"].tolist() for metadata_chunk in loaded_metadata]

    stacked_meta: np.ndarray = np.vstack(
        input_ids_collection,
    )
    stacked_meta: np.ndarray = stacked_meta.reshape(
        stacked_meta.shape[0] * stacked_meta.shape[1],
    )
    sentence_idx = np.array([np.ones(token_num) * i for i in range(sentence_num)]).reshape(sentence_num * token_num)
    full_df = pd.DataFrame(
        {
            "arr": list(array_np),
            "meta": list(stacked_meta),
            "sentence_idx": [int(x) for x in sentence_idx],
        },
    )

    return full_df


def remove_padding_and_extra_tokens(
    full_df: pd.DataFrame,
    main_config: MainConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> tuple[
    TransformersTokenizer,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Remove padding and extra tokens from the data."""
    tokenizer, _ = load_modified_tokenizer(
        main_config=main_config,
        logger=logger,
    )

    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"{eos_token_id = }",  # noqa: G004 - low overhead
        )
        logger.info(
            f"{pad_token_id = }",  # noqa: G004 - low overhead
        )

    if pad_token_id is None:
        msg = "The padding token id is None."
        raise ValueError(
            msg,
        )

    filtered_df = full_df[(full_df["meta"] != eos_token_id) & (full_df["meta"] != pad_token_id)]

    # arr_no_pad.shape:
    # (number of non-padding tokens in subsample, embedding dimension)
    arr_no_pad = np.array(
        list(filtered_df.arr),
    )

    # meta_no_pad.shape:
    # (number of non-padding tokens in subsample,)
    meta_no_pad = np.array(
        list(filtered_df.meta),
    )

    # sentence_idx_no_pad.shape:
    # (number of non-padding tokens in subsample,)
    sentence_idx_no_pad = np.array(
        list(filtered_df.sentence_idx),
    )

    return tokenizer, arr_no_pad, meta_no_pad, sentence_idx_no_pad


def select_subsets_of_arr_and_meta(
    arr_no_pad: np.ndarray,
    meta_no_pad: np.ndarray,
    sentence_idx_no_pad: np.ndarray,
    sample_size: int,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Select subsets of the arrays and metadata."""
    rng = np.random.default_rng(
        seed=42,
    )
    if len(arr_no_pad) >= sample_size:
        idx = rng.choice(
            range(len(arr_no_pad)),
            replace=False,
            size=sample_size,
        )
    else:
        idx = rng.choice(
            range(len(arr_no_pad)),
            replace=False,
            size=len(arr_no_pad),
        )

    arr_no_pad = arr_no_pad[idx]
    meta_no_pad = meta_no_pad[idx]
    sentence_idx_no_pad = sentence_idx_no_pad[idx]

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"{arr_no_pad.shape = }",  # noqa: G004 - low overhead
        )
        logger.info(
            f"Expected sample size: {sample_size = }",  # noqa: G004 - low overhead
        )

    return arr_no_pad, meta_no_pad, sentence_idx_no_pad


def save_prepared_data(
    embeddings_path_manager: EmbeddingsPathManager,
    arr_no_pad: np.ndarray,
    meta_frame: pd.DataFrame,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Save the prepared data."""
    prepared_data_dir_absolute_path = embeddings_path_manager.prepared_data_dir_absolute_path

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "prepared_data_dir_absolute_path:%s",
            prepared_data_dir_absolute_path,
        )

    # Make sure the save directory exists
    pathlib.Path(prepared_data_dir_absolute_path).mkdir(
        parents=True,
        exist_ok=True,
    )

    # # # #
    # Save the prepared data
    np.save(
        file=embeddings_path_manager.get_prepared_data_array_save_path(),
        arr=arr_no_pad,
    )

    meta_frame.to_pickle(
        path=embeddings_path_manager.get_prepared_data_meta_save_path(),
    )

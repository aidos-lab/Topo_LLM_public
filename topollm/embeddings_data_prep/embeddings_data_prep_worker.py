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

import numpy as np
import pandas as pd
import torch
import zarr

from topollm.config_classes.main_config import MainConfig
from topollm.embeddings_data_prep.load_pickle_files_from_meta_path import load_pickle_files_from_meta_path
from topollm.logging.log_dataframe_info import log_dataframe_info
from topollm.model_handling.tokenizer.load_tokenizer import load_modified_tokenizer
from topollm.path_management.embeddings.factory import get_embeddings_path_manager

logger = logging.getLogger(__name__)


def embeddings_data_prep_worker(
    main_config: MainConfig,
    device: torch.device,
    verbosity: int = 1,
    logger: logging.Logger = logger,
) -> None:
    """Prepare the embedding data of a model and its metadata for further analysis."""
    embeddings_path_manager = get_embeddings_path_manager(
        main_config=main_config,
        logger=logger,
    )

    # potentially adapt paths
    array_path = embeddings_path_manager.array_dir_absolute_path
    logger.info(
        "array_path",
        extra={
            "array_path": array_path,
        },
    )

    if not array_path.exists():
        msg = f"{array_path = } does not exist."
        raise FileNotFoundError(msg)

    partial_save_path = pathlib.Path(*list(array_path.parts)[-7:])
    prepared_save_path = pathlib.Path(
        "..",
        "..",
        "data",
        "analysis",
        "prepared",
        partial_save_path,
    )

    # Make sure the save path exists
    pathlib.Path(prepared_save_path).mkdir(
        parents=True,
        exist_ok=True,
    )

    meta_path = pathlib.Path(
        embeddings_path_manager.metadata_dir_absolute_path,
        "pickle_chunked_metadata_storage",
    )

    if verbosity >= 1:
        logger.info(f"{meta_path = }")

    if not meta_path.exists():
        msg = f"{meta_path = } does not exist."
        raise FileNotFoundError(msg)

    array = zarr.open(
        store=array_path,  # type: ignore
        mode="r",
    )

    arr = np.array(array)
    arr = arr.reshape(
        arr.shape[0] * arr.shape[1],
        arr.shape[2],
    )

    loaded_data = load_pickle_files_from_meta_path(
        meta_path=meta_path,
    )

    if verbosity >= 2:
        logger.info(
            "Loaded pickle files",
            extra={"loaded_data": loaded_data},
        )

    input_ids = []
    for i in range(len(loaded_data)):
        input_ids.append(loaded_data[i]["input_ids"].tolist())

    stacked_meta = np.vstack(input_ids)
    stacked_meta = stacked_meta.reshape(stacked_meta.shape[0] * stacked_meta.shape[1])

    stacked_meta_sub = stacked_meta

    full_df = pd.DataFrame(
        {
            "arr": list(arr),
            "meta": list(stacked_meta_sub),
        },
    )

    tokenizer, _ = load_modified_tokenizer(
        main_config=main_config,
        logger=logger,
    )

    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    if verbosity >= 1:
        logger.info(f"{eos_token_id = }")
        logger.info(f"{pad_token_id = }")

    if pad_token_id is None:
        msg = "The padding token id is None."
        raise ValueError(msg)

    # arr_no_pad.shape:
    # (number of non-padding tokens in subsample, embedding dimension)
    arr_no_pad = np.array(
        list(full_df[(full_df["meta"] != eos_token_id) & (full_df["meta"] != pad_token_id)].arr),
    )

    # meta_no_pad.shape:
    # (number of non-padding tokens in subsample,)
    meta_no_pad = np.array(
        list(full_df[(full_df["meta"] != eos_token_id) & (full_df["meta"] != pad_token_id)].meta),
    )
    # choose size of a meta sample which is used to take subsets for a point-wise
    # comparison of local estimators.
    np.random.seed(42)

    # Sample size of the arrays
    sample_size = main_config.embeddings_data_prep.num_samples

    if len(arr_no_pad) >= sample_size:
        idx = np.random.choice(
            range(len(arr_no_pad)),
            replace=False,
            size=sample_size,
        )
    else:
        idx = np.random.choice(
            range(len(arr_no_pad)),
            replace=False,
            size=len(arr_no_pad),
        )

    arr_no_pad = arr_no_pad[idx]
    meta_no_pad = meta_no_pad[idx]

    logger.info(f"Actual shape of the samples produced: {arr_no_pad.shape}")
    logger.info(f"Expected sample size: {sample_size}")

    file_name = f"embeddings_token_lvl_{sample_size}_samples_paddings_removed"
    np.save(
        file=pathlib.Path(
            prepared_save_path,
            file_name,
        ),
        arr=arr_no_pad,
    )

    # x of type 'numpy.int64' needs to be explicitly converted to an integer,
    # otherwise the convert_ids_to_tokens() method will raise the error:
    # TypeError: 'numpy.int64' object is not iterable
    token_names_no_pad = [tokenizer.convert_ids_to_tokens(int(x)) for x in meta_no_pad]

    meta_frame = pd.DataFrame(
        {
            "token_id": list(meta_no_pad),
            "token_name": list(token_names_no_pad),
        },
    )

    if verbosity >= 1:
        log_dataframe_info(
            meta_frame,
            df_name="meta_frame",
            logger=logger,
        )

    meta_name = f"{file_name}_meta.pkl"
    meta_frame.to_pickle(
        path=pathlib.Path(
            prepared_save_path,
            meta_name,
        ),
    )

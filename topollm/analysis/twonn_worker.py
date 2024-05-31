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
import torch
import zarr

from topollm.config_classes.main_config import MainConfig
from topollm.embeddings_data_prep.load_pickle_files_from_meta_path import load_pickle_files_from_meta_path
from topollm.logging.log_dataframe_info import log_dataframe_info
from topollm.model_handling.tokenizer.load_tokenizer import load_modified_tokenizer
from topollm.path_management.embeddings.factory import get_embeddings_path_manager

logger = logging.getLogger(__name__)


def twonn_worker(
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

    prepared_load_path = pathlib.Path(
        "..",
        "..",
        "data",
        "analysis",
        "prepared",
        partial_save_path,
    )

    prepared_save_path = pathlib.Path(
        "..",
        "..",
        "data",
        "analysis",
        "twonn",
        partial_save_path,
    )

    # Make sure the save path exists
    pathlib.Path(prepared_save_path).mkdir(
        parents=True,
        exist_ok=True,
    )

    if not prepared_load_path.exists():
        msg = f"{prepared_load_path = } does not exist."
        raise FileNotFoundError(msg)

    arr_no_pad = np.load(prepared_load_path)

    np.random.seed(2)
    sample_size = 1500
    sample_size = min(sample_size, arr_no_pad.shape[0])

    arr_no_pad = arr_no_pad[:sample_size]

    # provide number of jobs for the computation
    n_jobs = 1

    # provide number of neighbors which are used for the computation
    n_neighbors = round(len(arr_no_pad) * 0.8)

    print(arr_no_pad[:10])
    lPCA = skdim.id.TwoNN().fit_pw(arr_no_pad,
                                   n_neighbors=n_neighbors,
                                   n_jobs=n_jobs)

    ####

    array = list(lPCA.dimension_pw_)

    arr = np.array(array)

    file_name = f"twonn_token_lvl_{sample_size}_samples_paddings_removed"
    np.save(
        file=pathlib.Path(
            prepared_save_path,
            file_name,
        ),
        arr=arr,
    )

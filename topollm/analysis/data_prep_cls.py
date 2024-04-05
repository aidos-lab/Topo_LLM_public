# coding=utf-8
#
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

########################################################

# This is a script to prepare the embedding data of a
# model and its corresopnding fine-tuned variant.
# The script outputs two numpy arrays of subsamples
# of the respective arrays that correspond to the
# embeddings of the base model and the fine-tuned model,
# respectively.
# The arrays are stored in the directory where this
# script is executed.
# Since paddings are removed from the embeddings,
# the resulting size of the arrays will usually be
# significantly lower than the specified sample size
# (often ~5% of the specified size).

# third party imports
import pathlib
import hydra
import logging
import pathlib
import hydra
import omegaconf
import zarr
import numpy as np
import os
import pickle
import pandas as pd

from topollm.path_management.embeddings.EmbeddingsPathManagerFactory import (
    get_embeddings_path_manager,
)
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.config_classes.MainConfig import MainConfig
from topollm.logging.setup_exception_logging import setup_exception_logging

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Globals

# A logger for this file
global_logger = logging.getLogger(__name__)

setup_exception_logging(
    logger=global_logger,
)

# torch.set_num_threads(1)

# END Globals
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# function to load pickle files stored in the respective directory
def load_pickle_files(
    meta_path,
):
    data = []
    chunk_list = []

    for i in range(len(os.listdir(meta_path))):
        chunk_list.append(f"chunk_{str(i).zfill(5)}.pkl")

    for filename in chunk_list:
        if filename.endswith(".pkl"):
            filepath = os.path.join(
                meta_path,
                filename,
            )
            with open(filepath, "rb") as f:
                chunk = pickle.load(f)
                data.append(chunk)
    return data


@hydra.main(
    config_path="../../configs",
    config_name="main_config",
    version_base="1.2",
)
def main(
    config: omegaconf.DictConfig,
) -> None:
    main_config: MainConfig = initialize_configuration(
        config=config,
        logger=global_logger,
    )

    embeddings_path_manager = get_embeddings_path_manager(
        main_config=main_config,
        logger=global_logger,
    )

    # potentially adapt paths
    array_path = embeddings_path_manager.array_dir_absolute_path

    save_path = pathlib.Path(*list(array_path.parts)[-7:])
    save_path = pathlib.Path("..", "..", "data", "analysis", "prepared", save_path)

    meta_path = pathlib.Path(
        embeddings_path_manager.metadata_dir_absolute_path,
        "pickle_chunked_metadata_storage",
    )

    global_logger.info(f"{array_path = }")

    if not array_path.exists():
        raise FileNotFoundError(f"{array_path = } does not exist.")

    global_logger.info(f"{meta_path = }")

    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path = } does not exist.")

    array = zarr.open(
        store=array_path,  # type: ignore
        mode="r",
    )

    arr = np.array(array)
    arr = arr[:, 0, :]

    # choose sample size of the arrays
    sample_size = len(arr)
    np.random.seed(42)
    idx = np.random.choice(
        range(len(arr)),
        replace=False,
        size=sample_size,
    )

    arr_no_pad = arr[idx]

    print(arr_no_pad.shape)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_name = "embeddings_cls_lvl_" + str(sample_size) + "_samples_paddings_removed"
    np.save(
        pathlib.Path(save_path, file_name),
        arr_no_pad,
    )

    return None


if __name__ == "__main__":
    main()  # type: ignore

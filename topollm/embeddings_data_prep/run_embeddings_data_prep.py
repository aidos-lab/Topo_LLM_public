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

"""Prepare the embedding data of a model and its metadata for further analysis.

The script outputs two numpy arrays of subsamples
of the respective arrays that correspond to the
embeddings of the base model and the fine-tuned model,
respectively.
The arrays are stored in the directory where this
script is executed.
Since paddings are removed from the embeddings,
the resulting size of the arrays will usually be
significantly lower than the specified sample size
(often ~5% of the specified size).
"""

import logging
import os
import pathlib

import hydra
import hydra.core.hydra_config
import numpy as np
import omegaconf
import pandas as pd
import torch
import transformers
import zarr

from topollm.config_classes.MainConfig import MainConfig
from topollm.config_classes.setup_OmegaConf import setup_OmegaConf
from topollm.embeddings_data_prep.load_pickle_files_from_meta_path import load_pickle_files_from_meta_path
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.model_handling.get_torch_device import get_torch_device
from topollm.path_management.embeddings.EmbeddingsPathManagerFactory import (
    get_embeddings_path_manager,
)

# logger for this file
global_logger = logging.getLogger(__name__)

setup_exception_logging(
    logger=global_logger,
)

setup_OmegaConf()


@hydra.main(
    config_path="../../configs",
    config_name="main_config",
    version_base="1.2",
)
def main(
    config: omegaconf.DictConfig,
) -> None:
    """Run the script."""
    global_logger.info("Running script ...")

    main_config: MainConfig = initialize_configuration(
        config=config,
        logger=global_logger,
    )

    device = get_torch_device(
        preferred_torch_backend=main_config.preferred_torch_backend,
        logger=global_logger,
    )

    data_prep_worker(
        main_config=main_config,
        device=device,
        logger=global_logger,
    )


def data_prep_worker(
    main_config: MainConfig,
    device: torch.device,
    logger: logging.Logger,
) -> None:
    """Prepare the embedding data of a model and its metadata for further analysis."""
    embeddings_path_manager = get_embeddings_path_manager(
        main_config=main_config,
        logger=logger,
    )

    # potentially adapt paths
    array_path = embeddings_path_manager.array_dir_absolute_path
    logger.info(f"{array_path = }")

    if not array_path.exists():
        msg = f"{array_path = } does not exist."
        raise FileNotFoundError(msg)

    save_path = pathlib.Path(*list(array_path.parts)[-7:])
    save_path = pathlib.Path(
        "..",
        "..",
        "data",
        "analysis",
        "prepared",
        save_path,
    )

    meta_path = pathlib.Path(
        embeddings_path_manager.metadata_dir_absolute_path,
        "pickle_chunked_metadata_storage",
    )

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

    # choose size of a meta sample which is used to take subsets for a point-wise
    # comparison of local estimators.
    np.random.seed(42)

    meta_sample_size = 200000
    if meta_sample_size >= len(arr):
        idx = np.random.choice(
            range(len(arr)),
            replace=False,
            size=len(arr),
        )
    else:
        idx = np.random.choice(
            range(len(arr)),
            replace=False,
            size=meta_sample_size,
        )

    # Sample size of the arrays
    sample_size = main_config.embeddings_data_prep.num_samples

    idx = idx[:sample_size]
    arr = arr[idx]

    loaded_data = load_pickle_files_from_meta_path(
        meta_path=meta_path,
    )

    logger.info(f"Loaded pickle files:\n{loaded_data}")

    input_ids = []
    for i in range(len(loaded_data)):
        input_ids.append(loaded_data[i]["input_ids"].tolist())

    stacked_meta = np.vstack(input_ids)
    stacked_meta = stacked_meta.reshape(stacked_meta.shape[0] * stacked_meta.shape[1])

    stacked_meta_sub = stacked_meta[idx]

    df = pd.DataFrame(
        {
            "arr": list(arr),
            "meta": list(stacked_meta_sub),
        },
    )
    # ! TODO Currently, this hard-coded pad_token_id does not work for the GPT-2 tokenizer
    # TODO(Ben) Make this flexible so that we automatically extract the padding token index
    arr_no_pad = np.array(list(df[(df["meta"] != 2) & (df["meta"] != 1)].arr))
    meta_no_pad = np.array(list(df[(df["meta"] != 2) & (df["meta"] != 1)].meta))

    if not pathlib.Path(save_path).exists():
        os.makedirs(save_path)

    file_name = "embeddings_token_lvl_" + str(sample_size) + "_samples_paddings_removed"
    np.save(
        file=pathlib.Path(
            save_path,
            file_name,
        ),
        arr=arr_no_pad,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=main_config.language_model.pretrained_model_name_or_path,
    )
    token_names_no_pad = [tokenizer.decode(x) for x in meta_no_pad]

    meta_frame = pd.DataFrame(
        {
            "token_id": list(meta_no_pad),
            "token_name": list(token_names_no_pad),
        },
    )

    meta_name = f"{file_name}_meta.pkl"
    meta_frame.to_pickle(
        path=pathlib.Path(
            save_path,
            meta_name,
        ),
    )


if __name__ == "__main__":
    main()

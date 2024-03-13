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

"""
Create embedding vectors from dataset.
"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Imports

# Standard library imports
import logging
import os
import pathlib

# Third party imports
import hydra
import hydra.core.hydra_config
import omegaconf
import zarr


# Local imports
from topollm.config_classes.Configs import MainConfig
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging

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


@hydra.main(
    config_path="../../configs",
    config_name="main_config",
    version_base="1.2",
)
def main(
    config: omegaconf.DictConfig,
):
    """Run the script."""

    print("Running script ...")

    global_logger.info("Running script ...")

    main_config: MainConfig = initialize_configuration(
        config=config,
        logger=global_logger,
    )

    # # # #
    # Load the embeddings

    array_path = pathlib.Path(
        pathlib.Path.home(),
        "git-source",
        "Topo_LLM",
        "data",
        "embeddings",
        "arrays",
        "data-xsum_split-train_ctxt-dataset_entry/lvl-token/add-prefix-space-False_max-len-512/model-roberta-base_mask-no_masking/layer-[-1]_agg-mean/norm-None/",
        "array_dir",
        "test_array_dir",
    )

    if not array_path.exists():
        raise FileNotFoundError(f"{array_path = } does not exist.")

    array = zarr.open(
        store=array_path,  # type: ignore
        mode="r",
    )

    print(f"{array.shape = }")
    print(f"{array = }")
    print(f"{array[0] = }")

    # # # #
    # Load the metadata

    metadata_root_storage_path = pathlib.Path(
        pathlib.Path.home(),
        "git-source",
        "Topo_LLM",
        "data",
        "embeddings",
        "metadata",
        "data-xsum_split-train_ctxt-dataset_entry/lvl-token/add-prefix-space-False_max-len-512/model-roberta-base_mask-no_masking/layer-[-1]_agg-mean/norm-None/",
        "metadata_dir",
    )

    if not metadata_root_storage_path.exists():
        raise FileNotFoundError(f"{metadata_root_storage_path = } does not exist.")

    # "pickle_chunked_metadata_storage/chunk_00002.pkl"


if __name__ == "__main__":
    main()

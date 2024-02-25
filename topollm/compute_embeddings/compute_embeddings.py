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
import os
import pathlib
import pprint

# Third party imports
import datasets
import hydra
import hydra.core.hydra_config
import torch
import torch.utils.data
import zarr
from transformers import AutoModel

# Local imports
from topollm.config_classes.Configs import EmbeddingsConfig, DataConfig

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Globals

# A logger for this file
global_logger = logging.getLogger(__name__)

# END Globals
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@hydra.main(
    config_path="../../configs",
    config_name="config",
    version_base="1.2",
)
def main(
    config,
):
    """Run the script."""

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

    global_logger.info(
        f"embeddings_config:\n" f"{pprint.pformat(embeddings_config)}",
    )
    global_logger.info(
        f"data_config:\n" f"{pprint.pformat(data_config)}",
    )

    # Load the model
    model = AutoModel.from_pretrained(
        pretrained_model_name_or_path=embeddings_config.huggingface_model_name,
    )

    # Load the dataset from huggingface datasets
    dataset = datasets.load_dataset(
        data_config.dataset_identifier,
        trust_remote_code=True,
    )

    # TODO: Create split here
    # split=data_config.split,

    return


if __name__ == "__main__":
    main()

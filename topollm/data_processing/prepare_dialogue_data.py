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
Read ICLR text data from 'ICLR_Mistral_Embeddings.csv' and split to train/test/validation
"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Imports

# Standard library imports
import logging
import pathlib

# Third party imports
import hydra
import hydra.core.hydra_config
import omegaconf
import pandas as pd
from sklearn.model_selection import train_test_split

# Local imports
from topollm.config_classes.Configs import MainConfig
from topollm.logging.initialize_configuration_and_log import initialize_configuration
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


@hydra.main(
    config_path="../../configs",
    config_name="main_config",
    version_base="1.2",
)
def main(
    config: omegaconf.DictConfig,
) -> None:
    """Run the script."""

    print("Running script ...")

    global_logger.info("Running script ...")

    main_config: MainConfig = initialize_configuration(
        config=config,
        logger=global_logger,
    )

    # TODO Continue here

    return None


if __name__ == "__main__":
    main()

    global_logger.info("Script Done.")

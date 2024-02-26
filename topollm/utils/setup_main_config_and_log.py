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


# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Imports

# Standard library imports
import logging
import os
import pprint

# Third party imports
import hydra.core.hydra_config
import omegaconf

# Local imports

from topollm.config_classes.Configs import MainConfig
from topollm.utils.get_git_info import get_git_info

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def setup_main_config_and_log(
    config: omegaconf.DictConfig,
    logger: logging.Logger = logging.getLogger(__name__),
):
    """Sets up logging and validates the main configuration.

    Args:
        config:
            The configuration dictionary to be validated.

    Returns:
        The validated main configuration.
    """
    logger.info(f"{get_git_info() = }")

    main_config = MainConfig.model_validate(
        config,
    )

    if main_config.verbosity >= 1:
        logger.info(f"Working directory:\n" f"{os.getcwd() = }")
        logger.info(
            f"Hydra output directory:\n"
            f"{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}"
        )
        logger.info(
            f"hydra config:\n" f"{pprint.pformat(config)}",
        )
        logger.info(
            f"main_config:\n" f"{pprint.pformat(main_config)}",
        )

    return main_config

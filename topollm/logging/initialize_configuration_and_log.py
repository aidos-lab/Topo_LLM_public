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

"""Initialize the main configuration and log the configuration and git information."""

import logging
import pathlib
import pprint

import hydra.core.hydra_config
import omegaconf

from topollm.config_classes.main_config import MainConfig
from topollm.logging.get_git_info import get_git_info
from topollm.typing.enums import Verbosity

default_logger = logging.getLogger(__name__)


def setup_main_config(
    config: omegaconf.DictConfig,
) -> MainConfig:
    """Set up logging and validate the main configuration.

    Args:
    ----
        config:
            The configuration dictionary to be validated.

    Returns:
    -------
        The validated main configuration.

    """
    main_config = MainConfig.model_validate(
        obj=config,
    )

    return main_config


def log_hydra_main_config(
    config: omegaconf.DictConfig,
    main_config: MainConfig,
    logger: logging.Logger = default_logger,
) -> None:
    """Log the main configuration and the working directory."""
    logger.info(
        f"Working directory:\n{pathlib.Path.cwd() = }",  # noqa: G004 - low overhead
    )
    try:
        logger.info(
            "Hydra output directory:\n%s",
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
        )
    except ValueError:
        # If the config is not loaded with the proper hydra context manager,
        # it might lead to the following: `ValueError: HydraConfig was not set`
        # We catch this error and log a warning.
        logger.warning(
            "Hydra output directory could not be determined.",
        )
    logger.info(
        "omegaconf.DictConfig config:\n%s",
        pprint.pformat(config),
    )
    logger.info(
        "main_config:\n%s",
        pprint.pformat(main_config),
    )


def log_git_info(
    logger: logging.Logger = default_logger,
) -> None:
    """Log the git information."""
    logger.info(
        f"{get_git_info() = }",  # noqa: G004 - low overhead
    )


def initialize_configuration(
    config: omegaconf.DictConfig,
    logger: logging.Logger = default_logger,
) -> MainConfig:
    """Initialize the main configuration."""
    main_config = setup_main_config(
        config=config,
    )

    if main_config.verbosity >= Verbosity.NORMAL:
        log_git_info(
            logger=logger,
        )
        log_hydra_main_config(
            config=config,
            main_config=main_config,
            logger=logger,
        )

    return main_config

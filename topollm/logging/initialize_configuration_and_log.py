# Copyright 2024-2025
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_1 (author1@example.com)
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

"""Initialize the main configuration and log the configuration and git information."""

import logging
import pathlib
import pprint
import socket
import subprocess
import sys

import hydra.core.hydra_config
import omegaconf

from topollm.config_classes.main_config import MainConfig
from topollm.logging.get_git_info import get_git_info
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


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
    main_config: MainConfig = MainConfig.model_validate(
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
        msg=f"Working directory:\n{pathlib.Path.cwd() = }",  # noqa: G004 - low overhead
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
            msg="Hydra output directory could not be determined.",
        )
    logger.info(
        "omegaconf.DictConfig config:\n%s",
        pprint.pformat(object=config),
    )
    logger.info(
        "main_config:\n%s",
        pprint.pformat(object=main_config),
    )


def log_system_info(
    logger: logging.Logger = default_logger,
) -> None:
    """Log system hostname and environment information."""
    try:
        hostname = socket.gethostname()
    except Exception:  # noqa: BLE001 - We want to proceed no matter what the error is
        hostname = "unknown"

    logger.info(
        msg=f"Running on {hostname = }",  # noqa: G004 - low overhead
    )


def log_python_env_info(
    logger: logging.Logger = default_logger,
) -> None:
    """Log Python environment and Poetry information."""
    # Log Python version and executable
    try:
        python_version: str = sys.version.split()[0]
        python_path: str = sys.executable
        logger.info(
            msg=f"Python version: {python_version = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"Python executable: {python_path = }",  # noqa: G004 - low overhead
        )
    except Exception:  # noqa: BLE001 - We want to proceed no matter what the error is
        logger.info(msg="Unable to determine Python version/path")

    # Check Poetry environment
    try:
        result = subprocess.run(
            args=[
                "poetry",
                "env",
                "info",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            env_info: str = result.stdout.strip()
            logger.info(
                msg=f"Poetry environment:\n{env_info}",  # noqa: G004 - low overhead
            )
        else:
            logger.info(
                msg="Not running in a Poetry environment",
            )
    except Exception:  # noqa: BLE001 - We want to proceed no matter what the error is
        logger.info(
            msg="Unable to determine Poetry environment",
        )


def log_git_info(
    logger: logging.Logger = default_logger,
) -> None:
    """Log the git information."""
    logger.info(
        msg=f"{get_git_info() = }",  # noqa: G004 - low overhead
    )


def initialize_configuration(
    config: omegaconf.DictConfig,
    logger: logging.Logger = default_logger,
) -> MainConfig:
    """Initialize the main configuration."""
    main_config: MainConfig = setup_main_config(
        config=config,
    )

    if main_config.verbosity >= Verbosity.NORMAL:
        log_git_info(
            logger=logger,
        )
        log_system_info(
            logger=logger,
        )
        log_python_env_info(
            logger=logger,
        )
        log_hydra_main_config(
            config=config,
            main_config=main_config,
            logger=logger,
        )

    return main_config

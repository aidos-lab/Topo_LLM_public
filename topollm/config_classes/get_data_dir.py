# Copyright 2024-2025
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_1 (author1@example.com)
# AUTHOR_2 (author2@example.com)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#


import logging
import pathlib

from topollm.config_classes.main_config import MainConfig
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def get_data_dir(
    main_config: MainConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> pathlib.Path:
    """Get the data directory from the main configuration."""
    data_dir: pathlib.Path = main_config.paths.data_dir

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{data_dir = }",  # noqa: G004 - low overhead
        )

    return data_dir

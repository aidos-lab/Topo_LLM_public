"""Module with functions to retrieve the data directory from the main configuration."""

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

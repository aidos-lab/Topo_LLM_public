"""Factory for creating a finetuning path manager."""

import logging

from topollm.config_classes.main_config import MainConfig
from topollm.path_management.finetuning.finetuning_path_manager_basic import (
    FinetuningPathManagerBasic,
)
from topollm.path_management.finetuning.protocol import (
    FinetuningPathManager,
)

default_logger = logging.getLogger(__name__)


def get_finetuning_path_manager(
    main_config: MainConfig,
    logger: logging.Logger = default_logger,
) -> FinetuningPathManager:
    """Get a finetuning path manager based on the main configuration."""
    path_manger = FinetuningPathManagerBasic(
        data_config=main_config.data,
        paths_config=main_config.paths,
        finetuning_config=main_config.finetuning,
        verbosity=main_config.verbosity,
        logger=logger,
    )

    return path_manger

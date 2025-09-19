"""Factory for embeddings path managers."""

import logging

from topollm.config_classes.main_config import MainConfig
from topollm.path_management.embeddings.embeddings_path_manager_separate_directories import (
    EmbeddingsPathManagerSeparateDirectories,
)
from topollm.path_management.embeddings.protocol import (
    EmbeddingsPathManager,
)
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def get_embeddings_path_manager(
    main_config: MainConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> EmbeddingsPathManager:
    """Get an embeddings path manager based on the main configuration."""
    path_manger = EmbeddingsPathManagerSeparateDirectories(
        main_config=main_config,
        verbosity=verbosity,
        logger=logger,
    )

    return path_manger

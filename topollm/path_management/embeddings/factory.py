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

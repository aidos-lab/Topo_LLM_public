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


"""Setting up logging for the transformers library."""

from typing import TYPE_CHECKING

import transformers

if TYPE_CHECKING:
    import logging


def setup_transformers_logger() -> None:
    """Set up the transformers logger for propagating to the root logger."""
    # Set the transformers logging level
    transformers.logging.set_verbosity_info()

    # Make the transformers logger propagate to the root logger
    transformers_logger: logging.Logger = transformers.logging.get_logger()
    transformers_logger.handlers = []  # This avoids duplicate logging
    transformers_logger.propagate = True

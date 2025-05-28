# Copyright 2024
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

import transformers

from topollm.typing.enums import Verbosity

default_logger = logging.getLogger(__name__)


def save_tuned_model(
    trainer: transformers.Trainer,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Save the tuned model to disk."""
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "Calling trainer.save_model() ...",
        )

    trainer.save_model()

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "Calling trainer.save_model() DONE",
        )

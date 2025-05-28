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
import os
import pathlib

from topollm.path_management.finetuning.protocol import (
    FinetuningPathManager,
)

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def prepare_logging_dir(
    finetuning_path_manager: FinetuningPathManager,
    logger: logging.Logger = default_logger,
) -> pathlib.Path | None:
    """Prepare the logging directory for the finetuning process."""
    logging_dir: pathlib.Path | None = finetuning_path_manager.logging_dir
    logger.info(
        msg=f"{logging_dir = }",  # noqa: G004 - low overhead
    )

    # Create the logging directory if it does not exist
    if logging_dir is not None:
        logging_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
    else:
        logger.info(
            msg="No logging directory specified. Using default logging from transformers.Trainer.",
        )

    return logging_dir

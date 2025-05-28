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
from topollm.typing.enums import Verbosity

default_logger = logging.getLogger(__name__)


def prepare_finetuned_model_dir(
    finetuning_path_manager: FinetuningPathManager,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> pathlib.Path:
    """Prepare the directory for the finetuned model."""
    finetuned_model_dir: pathlib.Path = finetuning_path_manager.finetuned_model_dir

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"{finetuned_model_dir = }",  # noqa: G004 - low overhead
        )

    # Create the output directory if it does not exist
    finetuned_model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return finetuned_model_dir

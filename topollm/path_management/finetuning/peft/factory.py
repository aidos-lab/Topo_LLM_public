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


"""Factory for creating a PEFTPathManager instance."""

import logging

from topollm.config_classes.finetuning.peft.peft_config import PEFTConfig
from topollm.path_management.finetuning.peft.peft_path_manager_basic import (
    PEFTPathManagerBasic,
)
from topollm.path_management.finetuning.peft.protocol import (
    PEFTPathManager,
)
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def get_peft_path_manager(
    peft_config: PEFTConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> PEFTPathManager:
    """Create a PEFTPathManager instance."""
    path_manger = PEFTPathManagerBasic(
        peft_config=peft_config,
        verbosity=verbosity,
        logger=logger,
    )

    return path_manger

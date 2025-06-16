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


"""Protocol for managing the paths for the PEFT process."""

import pathlib
from typing import Protocol

from topollm.typing.enums import DescriptionType


class PEFTPathManager(Protocol):
    """Manages the paths for the PEFT process."""

    @property
    def peft_description_subdir(
        self,
    ) -> pathlib.Path: ...  # pragma: no cover

    def get_config_description(
        self,
        description_type: DescriptionType = DescriptionType.LONG,
        short_description_separator: str = "-",
    ) -> str: ...  # pragma: no cover

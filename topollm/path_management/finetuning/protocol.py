# Copyright 2024
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_1 (author1@example.com)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#


import pathlib
from typing import Protocol

from topollm.typing.enums import DescriptionType


class FinetuningPathManager(Protocol):
    """Manages the paths for the finetuning process."""

    def get_finetuned_model_relative_dir(
        self,
    ) -> pathlib.Path: ...  # pragma: no cover

    def get_finetuned_short_model_name(
        self,
    ) -> str: ...  # pragma: no cover

    def get_finetuning_parameters_description_for_short_model_name(
        self,
        short_description_separator: str = "-",
    ) -> str: ...  # pragma: no cover

    @property
    def finetuned_model_dir(
        self,
    ) -> pathlib.Path: ...  # pragma: no cover

    @property
    def logging_dir(
        self,
    ) -> pathlib.Path | None: ...  # pragma: no cover

# Copyright 2023-2024
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


"""Base class for dataclasses that should be hashable."""

from pydantic import BaseModel, ConfigDict


class BaseModelHashable(BaseModel):
    """Base class for dataclasses that should be hashable."""

    model_config = ConfigDict(
        frozen=True,
    )

    def __hash__(
        self,
    ) -> int:
        """Hash the object."""
        return hash(
            (type(self), *tuple(self.__dict__.values())),
        )

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


"""Protocol for modifying how the gradients behave during fune-tuning."""

from typing import Protocol

from topollm.typing.types import ModifiedModel


class GradientModifier(Protocol):
    """Modify the gradient behaviour during fine-tuning."""

    def modify_gradients(
        self,
        model: ModifiedModel,
    ) -> ModifiedModel: ...  # pragma: no cover

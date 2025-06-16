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


"""Protocol for model modifiers which can be used to influence the fine-tuning."""

from typing import Protocol

from transformers import PreTrainedModel

from topollm.typing.types import ModifiedModel


class ModelModifier(Protocol):
    """Modify a model for finetuning."""

    def modify_model(
        self,
        model: PreTrainedModel,
    ) -> ModifiedModel: ...  # pragma: no cover

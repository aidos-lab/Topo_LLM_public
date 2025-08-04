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


"""Model modifier that does not modify the model, i.e., leads to standard fine-tuning behavior."""

import logging

import peft.peft_model
from transformers import PreTrainedModel

from topollm.typing.enums import Verbosity

default_logger = logging.getLogger(__name__)


class ModelModifierStandard:
    """Model modifier that does not modify the model, i.e., leads to standard fine-tuning behavior."""

    def __init__(
        self,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Initialize the model modifier."""
        self.verbosity = verbosity
        self.logger = logger

    def modify_model(
        self,
        model: PreTrainedModel,
    ) -> peft.peft_model.PeftModel | PreTrainedModel:
        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info("Using base model without modifications.")
            self.logger.info("Returning unmodified model.")

        return model

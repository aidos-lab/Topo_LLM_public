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


"""Modify a tokenizer by doing nothing."""

import logging

from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


class TokenizerModifierDoNothing:
    """Modify a tokenizer by doing nothing."""

    def __init__(
        self,
        verbosity: int = 1,
        logger: logging.Logger = logger,
    ) -> None:
        self.verbosity = verbosity
        self.logger = logger

    def modify_tokenizer(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    ) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        if self.verbosity >= 1:
            self.logger.info("Returning unmodified tokenizer.")

        return tokenizer

    def update_model(
        self,
        model: PreTrainedModel,
    ) -> PreTrainedModel:
        if self.verbosity >= 1:
            self.logger.info("Returning unmodified model.")

        return model

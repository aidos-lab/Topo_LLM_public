"""Modify a tokenizer by doing nothing."""

import logging

from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from topollm.typing.enums import Verbosity

logger: logging.Logger = logging.getLogger(
    name=__name__,
)


class TokenizerModifierDoNothing:
    """Modify a tokenizer by doing nothing."""

    def __init__(
        self,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = logger,
    ) -> None:
        """Initialize the tokenizer modifier."""
        self.verbosity: Verbosity = verbosity
        self.logger: logging.Logger = logger

    def modify_tokenizer(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    ) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg="Returning unmodified tokenizer.",
            )

        return tokenizer

    def update_model(
        self,
        model: PreTrainedModel,
    ) -> PreTrainedModel:
        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info("Returning unmodified model.")

        return model

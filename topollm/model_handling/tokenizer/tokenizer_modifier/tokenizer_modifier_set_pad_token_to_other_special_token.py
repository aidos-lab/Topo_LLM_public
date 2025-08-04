"""Modify a tokenizer by setting the padding token to another special token."""

import logging

from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from topollm.typing.enums import Verbosity

logger: logging.Logger = logging.getLogger(
    name=__name__,
)


class TokenizerModifierSetPadTokenToOtherSpecialToken:
    """Modify a tokenizer by setting the padding token to another special token."""

    def __init__(
        self,
        other_special_token_identifier: str = "eos_token",  # noqa: S107 - this is not a password
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = logger,
    ) -> None:
        """Initialize the tokenizer modifier."""
        self.other_special_token_identifier: str = other_special_token_identifier

        self.verbosity: Verbosity = verbosity
        self.logger: logging.Logger = logger

    def modify_tokenizer(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    ) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        """Modify the tokenizer by setting the padding token to another special token."""
        # Check that the tokenizer has the other special token in its special tokens map
        if self.other_special_token_identifier not in tokenizer.special_tokens_map:
            msg: str = (
                f"Tokenizer does not have the special token '{self.other_special_token_identifier=}' "
                "in its special tokens map. Thus, cannot set the padding token to it."
            )
            raise ValueError(msg)
        # Check that the special token exists as an attribute of the tokenizer
        if not hasattr(
            tokenizer,
            self.other_special_token_identifier,
        ):
            msg: str = (
                f"Tokenizer does not have the special token '{self.other_special_token_identifier=}' "
                "as an attribute. Thus, cannot set the padding token to it."
            )
            raise ValueError(msg)

        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg=f"Setting padding token to special token "  # noqa: G004 - low overhead
                f"{tokenizer.special_tokens_map[self.other_special_token_identifier]=}.",
            )

        # Set the padding token to the other special token
        tokenizer.pad_token = getattr(
            tokenizer,
            self.other_special_token_identifier,
            # No default value here, as we want to raise an error if the token is not found
        )

        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg=f"Modified tokenizer {tokenizer = } by setting padding token to {tokenizer.pad_token = }.",  # noqa: G004 - low overhead
            )

        return tokenizer

    def update_model(
        self,
        model: PreTrainedModel,
    ) -> PreTrainedModel:
        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg="Returning unmodified model.",
            )

        return model

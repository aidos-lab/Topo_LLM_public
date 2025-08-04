"""Factory for tokenizer modifier."""

import logging

from topollm.config_classes.language_model.tokenizer_modifier.tokenizer_modifier_config import (
    TokenizerModifierConfig,
)
from topollm.model_handling.tokenizer.tokenizer_modifier import (
    protocol,
    tokenizer_modifier_add_padding_token,
    tokenizer_modifier_do_nothing,
    tokenizer_modifier_set_pad_token_to_other_special_token,
)
from topollm.typing.enums import TokenizerModifierMode, Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def get_tokenizer_modifier(
    tokenizer_modifier_config: TokenizerModifierConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> protocol.TokenizerModifier:
    """Get the tokenizer modifier based on the configuration."""
    mode: TokenizerModifierMode = tokenizer_modifier_config.mode

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{mode = }",  # noqa: G004 - low overhead
        )

    match mode:
        case TokenizerModifierMode.DO_NOTHING:
            modifier = tokenizer_modifier_do_nothing.TokenizerModifierDoNothing(
                verbosity=verbosity,
                logger=logger,
            )
        case TokenizerModifierMode.ADD_PADDING_TOKEN:
            if tokenizer_modifier_config.padding_token is None:
                msg: str = (
                    "TokenizerModifierMode.ADD_PADDING_TOKEN requires a padding token to be set. "
                    "Please provide a valid padding token in the configuration."
                )
                raise ValueError(
                    msg,
                )

            modifier = tokenizer_modifier_add_padding_token.TokenizerModifierAddPaddingToken(
                padding_token=tokenizer_modifier_config.padding_token,
                verbosity=verbosity,
                logger=logger,
            )
        case TokenizerModifierMode.REPLACE_PAD_TOKEN_WITH_OTHER_SPECIAL_TOKEN:
            modifier = tokenizer_modifier_set_pad_token_to_other_special_token.TokenizerModifierSetPadTokenToOtherSpecialToken(
                other_special_token_identifier=tokenizer_modifier_config.replace_pad_token_with_other_special_token_identifier,
                verbosity=verbosity,
                logger=logger,
            )
        case _:
            msg: str = f"Unknown {mode = }"
            raise ValueError(msg)

    return modifier

"""Interface function to load a tokenizer from a MainConfig object."""

import logging

from topollm.config_classes.main_config import MainConfig
from topollm.model_handling.tokenizer.load_tokenizer import load_modified_tokenizer
from topollm.model_handling.tokenizer.tokenizer_modifier.protocol import TokenizerModifier
from topollm.typing.enums import Verbosity
from topollm.typing.types import TransformersTokenizer

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def load_modified_tokenizer_from_main_config(
    main_config: MainConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> tuple[
    TransformersTokenizer,
    TokenizerModifier,
]:
    """Interface function to load a tokenizer from a MainConfig object."""
    (
        tokenizer,
        tokenizer_modifier,
    ) = load_modified_tokenizer(
        language_model_config=main_config.language_model,
        tokenizer_config=main_config.tokenizer,
        verbosity=verbosity,
        logger=logger,
    )

    return tokenizer, tokenizer_modifier

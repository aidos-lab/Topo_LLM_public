"""Load a tokenizer from a FinetuningConfig object."""

import logging

from topollm.config_classes.finetuning.finetuning_config import FinetuningConfig
from topollm.model_handling.tokenizer.load_tokenizer import load_modified_tokenizer
from topollm.model_handling.tokenizer.tokenizer_modifier.protocol import TokenizerModifier
from topollm.typing.enums import Verbosity
from topollm.typing.types import TransformersTokenizer

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def load_modified_tokenizer_from_finetuning_config(
    finetuning_config: FinetuningConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> tuple[
    TransformersTokenizer,
    TokenizerModifier,
]:
    """Interface function to load a tokenizer from a FinetuningConfig object."""
    (
        tokenizer,
        tokenizer_modifier,
    ) = load_modified_tokenizer(
        language_model_config=finetuning_config.base_model,
        tokenizer_config=finetuning_config.tokenizer,
        verbosity=verbosity,
        logger=logger,
    )

    # Make sure not to accidentally modify the tokenizer pad token (tokenizer.pad_token) here.
    # In particular, it is not custom to set the pad token to the eos token for masked language model training.

    return (
        tokenizer,
        tokenizer_modifier,
    )

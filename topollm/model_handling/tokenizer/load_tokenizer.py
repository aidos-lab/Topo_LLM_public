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


"""Load the tokenizer for a model."""

import logging
import os

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from topollm.config_classes.language_model.language_model_config import LanguageModelConfig
from topollm.config_classes.tokenizer.tokenizer_config import TokenizerConfig
from topollm.logging.log_tokenizer_info import log_tokenizer_info
from topollm.model_handling.tokenizer.tokenizer_modifier.factory import get_tokenizer_modifier
from topollm.model_handling.tokenizer.tokenizer_modifier.protocol import TokenizerModifier
from topollm.typing.enums import Verbosity
from topollm.typing.types import TransformersTokenizer

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def load_tokenizer(
    pretrained_model_name_or_path: str | os.PathLike,
    tokenizer_config: TokenizerConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    """Load the tokenizer based on the configuration."""
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Loading tokenizer {pretrained_model_name_or_path = } with "  # noqa: G004 - low overhead
            f"{tokenizer_config.add_prefix_space = } ...",
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            add_prefix_space=tokenizer_config.add_prefix_space,
        )
    except Exception as e:
        logger.exception(
            msg=f"Failed to load tokenizer {pretrained_model_name_or_path = } with "  # noqa: G004 - low overhead
            f"{tokenizer_config.add_prefix_space = }",
        )
        raise

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Loading tokenizer {pretrained_model_name_or_path = } with "  # noqa: G004 - low overhead
            f"{tokenizer_config.add_prefix_space = } DONE",
        )

    # Log the tokenizer information.
    if verbosity >= Verbosity.NORMAL:
        log_tokenizer_info(
            tokenizer=tokenizer,
            name="tokenizer",
            logger=logger,
        )

    return tokenizer


def load_modified_tokenizer(
    language_model_config: LanguageModelConfig,
    tokenizer_config: TokenizerConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> tuple[
    TransformersTokenizer,
    TokenizerModifier,
]:
    """Load the tokenizer and modify it if necessary."""
    if language_model_config.manual_tokenizer_override_pretrained_model_name_or_path is not None:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=">>> Note: Using manual tokenizer override. @@@",
            )
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = load_tokenizer(
            pretrained_model_name_or_path=language_model_config.manual_tokenizer_override_pretrained_model_name_or_path,
            tokenizer_config=tokenizer_config,
            verbosity=verbosity,
            logger=logger,
        )
    else:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Loading tokenizer without manual override.",
            )
        tokenizer = load_tokenizer(
            pretrained_model_name_or_path=language_model_config.pretrained_model_name_or_path,
            tokenizer_config=tokenizer_config,
            verbosity=verbosity,
            logger=logger,
        )

    tokenizer_modifier: TokenizerModifier = get_tokenizer_modifier(
        tokenizer_modifier_config=language_model_config.tokenizer_modifier,
        verbosity=verbosity,
        logger=logger,
    )

    tokenizer_modified: PreTrainedTokenizer | PreTrainedTokenizerFast = tokenizer_modifier.modify_tokenizer(
        tokenizer=tokenizer,
    )

    # Log the modified tokenizer information.
    if verbosity >= Verbosity.NORMAL:
        log_tokenizer_info(
            tokenizer=tokenizer_modified,
            name="tokenizer_modified",
            logger=logger,
        )

    return (
        tokenizer_modified,
        tokenizer_modifier,
    )

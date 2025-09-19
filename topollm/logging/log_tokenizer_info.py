"""Logging utilities for tokenizer information."""

import logging
import pprint

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)

default_logger_block_separator: str = "=" * 80
default_logger_section_separator: str = "-" * 80


def log_tokenizer_info(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    name: str = "tokenizer",
    logger_section_separator: str | None = default_logger_section_separator,
    logger_block_separator: str | None = default_logger_block_separator,
    logger: logging.Logger = default_logger,
) -> None:
    """Log model information."""
    if logger_block_separator is not None:
        logger.info(
            msg=logger_block_separator,
        )

    logger.info(
        msg=f"{name}:\n{tokenizer}",  # noqa: G004 - low overhead
    )

    if logger_section_separator is not None:
        logger.info(
            msg=logger_section_separator,
        )

    logger.info(
        f"{name}.__dict__:\n{pprint.pformat(object=tokenizer.__dict__)}",  # noqa: G004 - low overhead
    )

    if logger_block_separator is not None:
        logger.info(
            msg=logger_block_separator,
        )

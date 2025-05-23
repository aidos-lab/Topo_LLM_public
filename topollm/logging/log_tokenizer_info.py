# Copyright 2024-2025
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
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

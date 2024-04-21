# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (ruppik@hhu.de)
# Julius von Rohrscheidt (julius.rohrscheidt@helmholtz-muenchen.de)
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

"""Load the tokenizer for a model."""

import logging
import os

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from topollm.config_classes.main_config import MainConfig
from topollm.config_classes.tokenizer.tokenizer_config import TokenizerConfig
from topollm.model_handling.tokenizer.tokenizer_modifier.factory import get_tokenizer_modifier
from topollm.model_handling.tokenizer.tokenizer_modifier.protocol import TokenizerModifier

logger = logging.getLogger(__name__)


def load_tokenizer(
    pretrained_model_name_or_path: str | os.PathLike,
    tokenizer_config: TokenizerConfig,
    verbosity: int = 1,
    logger: logging.Logger = logger,
) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    """Load the tokenizer based on the configuration."""
    if verbosity >= 1:
        logger.info(f"Loading tokenizer {pretrained_model_name_or_path = } ...")

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        add_prefix_space=tokenizer_config.add_prefix_space,
    )

    if verbosity >= 1:
        logger.info(f"Loading tokenizer {pretrained_model_name_or_path = } DONE")
        logger.info(
            f"tokenizer:\n{tokenizer}",
        )

    return tokenizer


def load_modified_tokenizer(
    main_config: MainConfig,
    logger: logging.Logger = logger,
) -> tuple[
    PreTrainedTokenizer | PreTrainedTokenizerFast,
    TokenizerModifier,
]:
    """Load the tokenizer and modify it if necessary."""
    tokenizer = load_tokenizer(
        pretrained_model_name_or_path=main_config.language_model.pretrained_model_name_or_path,
        tokenizer_config=main_config.tokenizer,
        logger=logger,
        verbosity=main_config.verbosity,
    )
    tokenizer_modifier = get_tokenizer_modifier(
        tokenizer_modifier_config=main_config.language_model.tokenizer_modifier,
        verbosity=main_config.verbosity,
        logger=logger,
    )

    tokenizer_modified = tokenizer_modifier.modify_tokenizer(
        tokenizer=tokenizer,
    )

    return tokenizer_modified, tokenizer_modifier

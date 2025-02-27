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

"""Factory for tokenizer modifier."""

import logging

from topollm.config_classes.language_model.tokenizer_modifier.tokenizer_modifier_config import (
    TokenizerModifierConfig,
)
from topollm.model_handling.tokenizer.tokenizer_modifier import (
    protocol,
    tokenizer_modifier_add_padding_token,
    tokenizer_modifier_do_nothing,
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
            f"{mode = }",  # noqa: G004 - low overhead
        )

    if mode == TokenizerModifierMode.DO_NOTHING:
        modifier = tokenizer_modifier_do_nothing.TokenizerModifierDoNothing(
            verbosity=verbosity,
            logger=logger,
        )
    elif mode == TokenizerModifierMode.ADD_PADDING_TOKEN:
        modifier = tokenizer_modifier_add_padding_token.TokenizerModifierAddPaddingToken(
            padding_token=tokenizer_modifier_config.padding_token,
            verbosity=verbosity,
            logger=logger,
        )
    else:
        msg: str = f"Unknown {mode = }"
        raise ValueError(msg)

    return modifier

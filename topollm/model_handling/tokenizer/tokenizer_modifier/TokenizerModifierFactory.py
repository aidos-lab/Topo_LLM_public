# coding=utf-8
#
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

import logging

import topollm.model_handling.tokenizer.tokenizer_modifier.TokenizerModifierAddPaddingToken as TokenizerModifierAddPaddingToken
import topollm.model_handling.tokenizer.tokenizer_modifier.TokenizerModifierDoNothing as TokenizerModifierDoNothing
import topollm.model_handling.tokenizer.tokenizer_modifier.TokenizerModifierProtocol as TokenizerModifierProtocol
from topollm.config_classes.enums import TokenizerModifierMode
from topollm.config_classes.finetuning.TokenizerModifierConfig import (
    TokenizerModifierConfig,
)


def get_tokenizer_modifier(
    tokenizer_modifier_config: TokenizerModifierConfig,
    verbosity: int = 1,
    logger: logging.Logger = logging.getLogger(__name__),
) -> TokenizerModifierProtocol.TokenizerModifier:
    mode: TokenizerModifierMode = tokenizer_modifier_config.mode

    if verbosity >= 1:
        logger.info(f"{mode = }")

    if mode == TokenizerModifierMode.DO_NOTHING:
        modifier = TokenizerModifierDoNothing.TokenizerModifierDoNothing(
            verbosity=verbosity,
            logger=logger,
        )
    elif mode == TokenizerModifierMode.ADD_PADDING_TOKEN:
        modifier = TokenizerModifierAddPaddingToken.TokenizerModifierAddPaddingToken(
            padding_token=tokenizer_modifier_config.padding_token,
            verbosity=verbosity,
            logger=logger,
        )
    else:
        raise ValueError(f"Unknown " f"{mode = }")

    return modifier

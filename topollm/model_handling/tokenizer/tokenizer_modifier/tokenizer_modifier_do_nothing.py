# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Matthias Ruppik (mail@ruppik.net)
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

"""Modify a tokenizer by doing nothing."""

import logging

from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


class TokenizerModifierDoNothing:
    """Modify a tokenizer by doing nothing."""

    def __init__(
        self,
        verbosity: int = 1,
        logger: logging.Logger = logger,
    ) -> None:
        self.verbosity = verbosity
        self.logger = logger

    def modify_tokenizer(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    ) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        if self.verbosity >= 1:
            self.logger.info("Returning unmodified tokenizer.")

        return tokenizer

    def update_model(
        self,
        model: PreTrainedModel,
    ) -> PreTrainedModel:
        if self.verbosity >= 1:
            self.logger.info("Returning unmodified model.")

        return model

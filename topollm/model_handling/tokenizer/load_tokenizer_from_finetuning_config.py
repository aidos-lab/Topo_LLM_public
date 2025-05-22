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

"""Load a tokenizer from a FinetuningConfig object."""

import logging

from topollm.config_classes.finetuning.finetuning_config import FinetuningConfig
from topollm.model_handling.tokenizer.load_tokenizer import load_modified_tokenizer
from topollm.model_handling.tokenizer.tokenizer_modifier.protocol import TokenizerModifier
from topollm.typing.enums import Verbosity
from topollm.typing.types import TransformersTokenizer

default_logger = logging.getLogger(__name__)


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

    return tokenizer, tokenizer_modifier

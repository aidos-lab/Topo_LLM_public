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
import os

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from topollm.config_classes.tokenizer.TokenizerConfig import TokenizerConfig


def load_tokenizer(
    pretrained_model_name_or_path: str | os.PathLike,
    tokenizer_config: TokenizerConfig,
    verbosity: int = 1,
    logger: logging.Logger = logging.getLogger(__name__),
) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    """
    Loads the tokenizer and model based on the configuration,
    and puts the model in evaluation mode.

    Args:
        pretrained_model_name_or_path:
            The name or path of the pretrained model.

    """
    if verbosity >= 1:
        logger.info(f"Loading tokenizer " f"{pretrained_model_name_or_path = } ...")

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        add_prefix_space=tokenizer_config.add_prefix_space,
    )

    if verbosity >= 1:
        logger.info(f"Loading tokenizer " f"{pretrained_model_name_or_path = } DONE")
        logger.info(
            f"tokenizer:\n" f"{tokenizer}",
        )

    return tokenizer

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

import transformers

from topollm.config_classes.finetuning.FinetuningConfig import FinetuningConfig
from topollm.model_handling.load_tokenizer import load_tokenizer


def load_tokenizer_from_FinetuningConfig(
    finetuning_config: FinetuningConfig,
    verbosity: int = 1,
    logger: logging.Logger = logging.getLogger(__name__),
) -> transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast:

    tokenizer = load_tokenizer(
        pretrained_model_name_or_path=finetuning_config.pretrained_model_name_or_path,
        tokenizer_config=finetuning_config.tokenizer,
        verbosity=verbosity,
        logger=logger,
    )

    # Make sure not to accidentally modify the tokenizer pad token (tokenizer.pad_token) here.
    # In particular, it is not custom to set the pad token to the eos token for masked language model training.

    return tokenizer

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
from transformers import AutoTokenizer

from topollm.config_classes.FinetuningConfig import FinetuningConfig


def load_tokenizer(
    finetuning_config: FinetuningConfig,
    logger: logging.Logger = logging.getLogger(__name__),
) -> transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast:
    # ? Do we need other config arguments for the tokenizer for finetuning here?

    logger.info(
        f"Loading tokenizer "
        f"{finetuning_config.pretrained_model_name_or_path = } ..."
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=finetuning_config.pretrained_model_name_or_path,
    )
    logger.info(
        f"Loading tokenizer "
        f"{finetuning_config.pretrained_model_name_or_path = } DONE"
    )
    logger.info(f"tokenizer:\n{tokenizer}")

    # Make sure not to accidentally modify the tokenizer pad token (tokenizer.pad_token) here.
    # In particular, it is not custom to set the pad token to the eos token for masked language model training.

    return tokenizer

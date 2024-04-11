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

from topollm.config_classes.enums import LMmode
from topollm.config_classes.finetuning.FinetuningConfig import FinetuningConfig


def prepare_data_collator(
    finetuning_config: FinetuningConfig,
    tokenizer: transformers.PreTrainedTokenizerBase,
    verbosity: int = 1,
    logger: logging.Logger = logging.getLogger(__name__),
):
    lm_mode = finetuning_config.lm_mode

    if lm_mode == LMmode.MLM:
        mlm = True
    elif lm_mode == LMmode.CLM:
        mlm = False
    else:
        raise ValueError(f"Unknown LMmode: " f"{lm_mode = }")

    data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=mlm,
        mlm_probability=finetuning_config.mlm_probability,
    )

    if verbosity >= 1:
        logger.info(f"{lm_mode = }")
        logger.info(f"{mlm = }")
        logger.info(f"{data_collator = }")

    return data_collator

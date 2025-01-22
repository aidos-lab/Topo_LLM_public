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

"""Prepare the data collator for the finetuning process."""

import logging

import transformers

from topollm.config_classes.finetuning.finetuning_config import FinetuningConfig
from topollm.typing.enums import LMmode, TaskType, Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def prepare_data_collator(
    finetuning_config: FinetuningConfig,
    tokenizer: transformers.PreTrainedTokenizerBase,
    verbosity: int = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> transformers.DataCollatorForLanguageModeling | transformers.DataCollatorForTokenClassification:
    """Prepare the data collator for the finetuning process."""
    match finetuning_config.base_model.task_type:
        case TaskType.MASKED_LM:
            data_collator = transformers.DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=True,
                mlm_probability=finetuning_config.mlm_probability,
            )
        case TaskType.CAUSAL_LM:
            data_collator = transformers.DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            )
        case TaskType.TOKEN_CLASSIFICATION:
            data_collator = transformers.DataCollatorForTokenClassification(
                tokenizer=tokenizer,
            )
        case _:
            msg: str = f"Unknown {finetuning_config.base_model.task_type = }"
            raise ValueError(
                msg,
            )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{data_collator = }",  # noqa: G004 - low overhead
        )

    return data_collator

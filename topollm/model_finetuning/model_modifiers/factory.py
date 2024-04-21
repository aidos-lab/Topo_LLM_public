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

import torch

from topollm.config_classes.enums import FinetuningMode
from topollm.config_classes.finetuning.peft.peft_config import PEFTConfig
from topollm.config_classes.finetuning.peft.peft_config_to_lora_config import (
    peft_config_to_lora_config,
)
from topollm.model_finetuning.model_modifiers import ModelModifierLora, ModelModifierStandard
from topollm.model_finetuning.model_modifiers.ModelModifierProtocol import ModelModifier

logger = logging.getLogger(__name__)


def get_model_modifier(
    peft_config: PEFTConfig,
    device: torch.device,
    verbosity: int = 1,
    logger: logging.Logger = logger,
) -> ModelModifier:
    finetuning_mode = peft_config.finetuning_mode
    if verbosity >= 1:
        logger.info(f"{finetuning_mode = }")

    if finetuning_mode == FinetuningMode.STANDARD:
        model_modifier = ModelModifierStandard.ModelModifierStandard(
            logger=logger,
        )
    elif finetuning_mode == FinetuningMode.LORA:
        lora_config = peft_config_to_lora_config(
            peft_config=peft_config,
        )

        if verbosity >= 1:
            logger.info(f"Preparing LoRA adapter ...")
            logger.info(f"{lora_config = }")

        model_modifier = ModelModifierLora.ModelModifierLora(
            lora_config=lora_config,
            device=device,
            verbosity=verbosity,
            logger=logger,
        )

        if verbosity >= 1:
            logger.info(f"Preparing LoRA adapter DONE.")
    else:
        raise ValueError(f"Unknown training mode: " f"{finetuning_mode = }")

    return model_modifier

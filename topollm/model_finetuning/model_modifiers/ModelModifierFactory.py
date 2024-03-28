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

import torch

import topollm.model_finetuning.model_modifiers.ModelModifierLora as ModelModifierLora
import topollm.model_finetuning.model_modifiers.ModelModifierStandard as ModelModifierStandard
from topollm.config_classes.enums import FinetuningMode
from topollm.config_classes.finetuning.peft.PEFTConfig import PEFTConfig
from topollm.config_classes.finetuning.peft.PEFTConfig_to_LoraConfig import (
    PEFTConfig_to_LoraConfig,
)
from topollm.model_finetuning.model_modifiers.ModelModifierProtocol import ModelModifier


def get_model_modifier(
    peft_config: PEFTConfig,
    device: torch.device,
    logger: logging.Logger = logging.getLogger(__name__),
) -> ModelModifier:
    finetuning_mode = peft_config.finetuning_mode
    logger.info(f"{finetuning_mode = }")

    if finetuning_mode == FinetuningMode.STANDARD:
        model_modifier = ModelModifierStandard.ModelModifierStandard(
            logger=logger,
        )
    elif finetuning_mode == FinetuningMode.LORA:
        lora_config = PEFTConfig_to_LoraConfig(
            peft_config=peft_config,
        )
        logger.info(f"{lora_config = }")

        model_modifier = ModelModifierLora.ModelModifierLora(
            lora_config=lora_config,
            logger=logger,
        )

        logger.info(f"Preparing LoRA adapter DONE.")
    else:
        raise ValueError(f"Unknown training mode: " f"{finetuning_mode = }")

    return model_modifier

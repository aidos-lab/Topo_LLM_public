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

from topollm.config_classes.finetuning.peft.peft_config import PEFTConfig
from topollm.config_classes.finetuning.peft.peft_config_to_lora_config import (
    peft_config_to_lora_config,
)
from topollm.model_finetuning.model_modifiers import model_modifier_lora, model_modifier_standard
from topollm.model_finetuning.model_modifiers.protocol import ModelModifier
from topollm.typing.enums import FinetuningMode, Verbosity

default_logger = logging.getLogger(__name__)


def get_model_modifier(
    peft_config: PEFTConfig,
    device: torch.device,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> ModelModifier:
    finetuning_mode = peft_config.finetuning_mode
    if verbosity >= 1:
        logger.info(f"{finetuning_mode = }")  # noqa: G004 - low overhead

    if finetuning_mode == FinetuningMode.STANDARD:
        model_modifier = model_modifier_standard.ModelModifierStandard(
            logger=logger,
        )
    elif finetuning_mode == FinetuningMode.LORA:
        lora_config = peft_config_to_lora_config(
            peft_config=peft_config,
        )

        if verbosity >= 1:
            logger.info("Preparing LoRA adapter ...")
            logger.info(
                "lora_config:\n%s",
                lora_config,
            )

        model_modifier = model_modifier_lora.ModelModifierLora(
            lora_config=lora_config,
            device=device,
            verbosity=verbosity,
            logger=logger,
        )

        if verbosity >= 1:
            logger.info("Preparing LoRA adapter DONE.")
    else:
        msg = f"Unknown training mode: {finetuning_mode = }"
        raise ValueError(msg)

    return model_modifier

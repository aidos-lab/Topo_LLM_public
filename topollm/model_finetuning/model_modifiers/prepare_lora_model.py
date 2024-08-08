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

import peft.mapping
import peft.peft_model
import torch
from peft.tuners.lora.config import LoraConfig
from transformers import PreTrainedModel

from topollm.logging.log_model_info import log_model_info


def prepare_lora_model(
    base_model: PreTrainedModel,
    lora_config: LoraConfig,
    device: torch.device,
    verbosity: int = 1,
    logger: logging.Logger = logging.getLogger(__name__),
) -> peft.peft_model.PeftModel:
    if verbosity >= 1:
        logger.info(f"Preparing LoRA adapter ...")
        logger.info(f"base_model before modification:")
        log_model_info(
            model=base_model,
            model_name="base_model",
            logger=logger,
        )

    # Get the model prepared with PEFT
    # (here: LoRA)
    lora_model = peft.mapping.get_peft_model(
        model=base_model,
        peft_config=lora_config,
        adapter_name="default",
    )
    lora_model.print_trainable_parameters()

    if not isinstance(
        lora_model,
        peft.peft_model.PeftModel,
    ):
        raise ValueError(f"Expected peft.peft_model.PeftModel, " f"but got {type(lora_model) = }")

    if verbosity >= 1:
        logger.info(f"lora_model after modification:")
        log_model_info(
            model=lora_model,
            model_name="lora_model",
            logger=logger,
        )

    if verbosity >= 1:
        logger.info(f"Moving model to {device = } ...")
    lora_model.to(
        device,  # type: ignore
    )
    if verbosity >= 1:
        logger.info(f"Moving model to {device = } DONE")

    return lora_model

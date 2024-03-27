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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Imports

# Standard library imports
import logging

# Third party imports
import peft.mapping
import peft.peft_model
import torch
from peft.tuners.lora.config import LoraConfig
from transformers import PreTrainedModel

# Local imports

from topollm.logging.log_model_info import log_model_info

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def prepare_lora_model(
    base_model: PreTrainedModel,
    lora_config: LoraConfig,
    device: torch.device,
    logger: logging.Logger = logging.getLogger(__name__),
) -> peft.peft_model.PeftModel:

    # Get the model prepared with PEFT (LoRA + LOFT-Q)
    lora_model = peft.mapping.get_peft_model(
        model=base_model,
        peft_config=lora_config,
        adapter_name="default",
    )
    lora_model.print_trainable_parameters()

    assert isinstance(
        lora_model,
        peft.peft_model.PeftModel,
    )

    log_model_info(
        model=lora_model,
        logger=logger,
    )

    # Move the model to GPU if available
    logger.info(f"Moving model to {device = } ...")
    lora_model.to(
        device,  # type: ignore
    )
    logger.info(f"Moving model to {device = } DONE")

    return lora_model

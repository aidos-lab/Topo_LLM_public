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

import pytest
import torch
from peft import PeftModel
from peft.tuners.lora.config import LoraConfig
from transformers import PreTrainedModel

from topollm.model_finetuning.model_modifiers.prepare_lora_model import (
    prepare_lora_model,
)

logger: logging.Logger = logging.getLogger(
    name=__name__,
)


@pytest.mark.uses_transformers_models
def test_prepare_lora_model_integration(
    base_model: PreTrainedModel,
    lora_config: LoraConfig,
    device_fixture: torch.device,
    logger: logging.Logger = logger,
) -> None:
    """Test the integration of the prepare_lora_model function with a real model and LoRA configuration."""
    modified_model: PeftModel = prepare_lora_model(
        base_model=base_model,
        lora_config=lora_config,
        device=device_fixture,
        logger=logger,
    )

    # Assertions to validate integration
    assert modified_model is not None, "The modified model should not be None"  # noqa: S101 - pytest assertion

    # You can add more specific assertions here depending on the expected behavior,
    # such as checking for the addition of specific LoRA parameters
    # or changes in parameter count.

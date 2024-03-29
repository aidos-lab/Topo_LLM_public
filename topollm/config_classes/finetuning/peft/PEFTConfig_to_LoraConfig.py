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

from topollm.config_classes.finetuning.peft.PEFTConfig import PEFTConfig


from peft.tuners.lora.config import LoraConfig


def PEFTConfig_to_LoraConfig(
    peft_config: PEFTConfig,
) -> LoraConfig:
    """
    Convert a PEFTConfig to a LoraConfig.

    https://huggingface.co/docs/peft/v0.10.0/en/package_reference/lora#peft.LoraConfig
    """

    # Note: The 'task_type' argument is not necessary.
    # task_type=peft.utils.peft_types.TaskType.CAUSAL_LM
    lora_config = LoraConfig(
        r=peft_config.r,
        lora_alpha=peft_config.lora_alpha,
        target_modules=peft_config.target_modules,
        lora_dropout=peft_config.lora_dropout,
    )

    return lora_config

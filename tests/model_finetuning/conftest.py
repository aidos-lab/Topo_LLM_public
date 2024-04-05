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

import pytest
import torch
from peft.tuners.lora.config import LoraConfig
from transformers import AutoModel


@pytest.fixture(
    scope="session",
)
def base_model():
    """
    Load a lightweight model for testing.
    """
    base_model = AutoModel.from_pretrained(
        "google/bert_uncased_L-2_H-128_A-2",
        torchscript=True,
    )

    return base_model


@pytest.fixture(
    scope="session",
)
def device():
    # Use a simple device selection for testing
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(
    scope="session",
)
def lora_config():
    # Create a test LoRA configuration. Adjust parameters as needed.
    config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.01,
        target_modules=[
            "query",
            "key",
            "value",
        ],
    )

    return config

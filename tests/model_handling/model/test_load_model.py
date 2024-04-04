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
import os

import pytest
from transformers import PreTrainedModel
import torch

from tests.model_handling.parameter_lists import (
    example_pretrained_model_name_or_path_list,
)
from topollm.model_handling.model.load_model import load_model


@pytest.mark.parametrize(
    "pretrained_model_name_or_path",
    example_pretrained_model_name_or_path_list,
)
@pytest.mark.uses_transformers_models
def test_load_model(
    pretrained_model_name_or_path: str | os.PathLike,
    device_fixture: torch.device,
    logger_fixture: logging.Logger,
) -> None:
    model = load_model(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        device=device_fixture,
        verbosity=1,
        logger=logger_fixture,
    )

    assert model is not None
    assert isinstance(
        model,
        PreTrainedModel,
    )

    return None

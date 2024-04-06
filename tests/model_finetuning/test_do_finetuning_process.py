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

import pytest
import torch

from topollm.config_classes.MainConfig import MainConfig
from topollm.model_finetuning.do_finetuning_process import do_finetuning_process


@pytest.mark.uses_transformers_models
@pytest.mark.high_memory_usage
@pytest.mark.slow
@pytest.mark.very_slow
def test_do_finetuning_process(
    main_config: MainConfig,
    device_fixture: torch.device,
    logger_fixture: logging.Logger,
) -> None:

    do_finetuning_process(
        main_config=main_config,
        device=device_fixture,
        logger=logger_fixture,
    )

    return None

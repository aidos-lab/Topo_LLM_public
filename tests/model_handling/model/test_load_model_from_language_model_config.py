# Copyright 2024-2025
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Matthias Ruppik (mail@ruppik.net)
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
from transformers import PreTrainedModel

from topollm.config_classes.language_model.language_model_config import LanguageModelConfig
from topollm.model_handling.model.load_model_from_language_model_config import load_model_from_language_model_config
from topollm.typing.enums import Verbosity


@pytest.mark.uses_transformers_models
def test_load_model(
    language_model_config: LanguageModelConfig,
    device_fixture: torch.device,
    verbosity: Verbosity,
    logger_fixture: logging.Logger,
    from_pretrained_kwargs_instance: dict | None = None,
) -> None:
    """Test the load_model function."""
    model: PreTrainedModel = load_model_from_language_model_config(
        language_model_config=language_model_config,
        from_pretrained_kwargs_instance=from_pretrained_kwargs_instance,
        device=device_fixture,
        verbosity=verbosity,
        logger=logger_fixture,
    )

    logger_fixture.info(
        msg=f"Loaded model:\n{model}",  # noqa: G004 - low overhead
    )

    assert model is not None  # noqa: S101 - pytest assert
    assert isinstance(  # noqa: S101 - pytest assert
        model,
        PreTrainedModel,
    )

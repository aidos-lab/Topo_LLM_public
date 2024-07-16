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
from pydantic import BaseModel
from transformers import PreTrainedModel

from topollm.config_classes.finetuning.finetuning_config import FinetuningConfig
from topollm.model_handling.model.load_model_from_language_model_config import load_model_from_language_model_config
from topollm.typing.enums import Verbosity

default_device = torch.device("cpu")
default_logger = logging.getLogger(__name__)


def load_base_model_from_finetuning_config(
    finetuning_config: FinetuningConfig,
    from_pretrained_kwargs_instance: BaseModel | dict | None = None,
    device: torch.device = default_device,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> PreTrainedModel:
    """Interface function to load a model from a FinetuningConfig object."""
    language_model_config = finetuning_config.base_model

    model = load_model_from_language_model_config(
        language_model_config=language_model_config,
        from_pretrained_kwargs_instance=from_pretrained_kwargs_instance,
        device=device,
        verbosity=verbosity,
        logger=logger,
    )

    return model

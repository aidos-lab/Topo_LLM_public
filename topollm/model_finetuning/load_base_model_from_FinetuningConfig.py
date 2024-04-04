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

import torch
from transformers import PreTrainedModel

from topollm.config_classes.finetuning.FinetuningConfig import FinetuningConfig
from topollm.model_handling.model.load_model import load_model


def load_base_model_from_FinetuningConfig(
    finetuning_config: FinetuningConfig,
    device: torch.device = torch.device("cpu"),
    logger: logging.Logger = logging.getLogger(__name__),
) -> PreTrainedModel:
    """
    Interface function to load a model from a FinetuningConfig object.
    """

    model = load_model(
        pretrained_model_name_or_path=finetuning_config.pretrained_model_name_or_path,
        device=device,
        verbosity=1,
        logger=logger,
    )

    return model

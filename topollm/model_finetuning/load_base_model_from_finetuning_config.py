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
import transformers
from transformers import PreTrainedModel

from topollm.config_classes.enums import LMmode
from topollm.config_classes.finetuning.finetuning_config import FinetuningConfig
from topollm.model_handling.model.load_model import load_model

default_device = torch.device("cpu")
logger = logging.getLogger(__name__)


def load_base_model_from_finetuning_config(
    finetuning_config: FinetuningConfig,
    device: torch.device = default_device,
    verbosity: int = 1,
    logger: logging.Logger = logger,
) -> PreTrainedModel:
    """Interface function to load a model from a FinetuningConfig object."""
    lm_mode = finetuning_config.lm_mode

    if lm_mode == LMmode.MLM:
        model_loading_class = transformers.AutoModelForMaskedLM
    elif lm_mode == LMmode.CLM:
        model_loading_class = transformers.AutoModelForCausalLM
    else:
        msg = f"Invalid lm_mode: {lm_mode = }"
        raise ValueError(msg)

    model = load_model(
        pretrained_model_name_or_path=finetuning_config.pretrained_model_name_or_path,
        model_loading_class=model_loading_class,
        device=device,
        verbosity=verbosity,
        logger=logger,
    )

    return model

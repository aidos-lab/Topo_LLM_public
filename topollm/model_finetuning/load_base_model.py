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
from transformers import AutoModelForMaskedLM, PreTrainedModel

from topollm.config_classes.finetuning.FinetuningConfig import FinetuningConfig
from topollm.logging.log_model_info import log_model_info


def load_base_model(
    finetuning_config: FinetuningConfig,
    device: torch.device = torch.device("cpu"),
    logger: logging.Logger = logging.getLogger(__name__),
) -> PreTrainedModel:
    logger.info(
        f"Loading model " f"{finetuning_config.pretrained_model_name_or_path = } ..."
    )
    model = AutoModelForMaskedLM.from_pretrained(
        pretrained_model_name_or_path=finetuning_config.pretrained_model_name_or_path,
    )
    logger.info(
        f"Loading model " f"{finetuning_config.pretrained_model_name_or_path = } DONE"
    )

    # Check type of model
    assert isinstance(
        model,
        PreTrainedModel,
    )

    log_model_info(
        model=model,
        logger=logger,
    )

    # Move the model to GPU if available
    logger.info(f"Moving model to {device = } ...")
    model.to(
        device,  # type: ignore
    )
    logger.info(f"Moving model to {device = } DONE")

    return model

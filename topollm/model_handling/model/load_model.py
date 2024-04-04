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

import torch
import transformers
from transformers import AutoModelForPreTraining, PreTrainedModel

from topollm.logging.log_model_info import log_model_info


def load_model(
    pretrained_model_name_or_path: str | os.PathLike,
    model_loading_class: type = transformers.AutoModelForPreTraining,
    device: torch.device = torch.device("cpu"),
    verbosity: int = 1,
    logger: logging.Logger = logging.getLogger(__name__),
) -> transformers.PreTrainedModel:
    """
    Loads the model based on the configuration.

    Args:
        pretrained_model_name_or_path:
            The name or path of the pretrained model.
    """

    if verbosity >= 1:
        logger.info(f"Loading model " f"{pretrained_model_name_or_path = } ...")

    if not hasattr(
        model_loading_class,
        "from_pretrained",
    ):
        raise ValueError(
            f"model_loading_class does not have a from_pretrained method: "
            f"{model_loading_class = }",
        )

    model: transformers.PreTrainedModel = model_loading_class.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
    )
    if verbosity >= 1:
        logger.info(f"Loading model " f"{pretrained_model_name_or_path = } DONE")

    if not isinstance(
        model,
        transformers.PreTrainedModel,
    ):
        raise ValueError(
            f"model is not of type PreTrainedModel: " f"{type(model) = }",
        )

    if verbosity >= 1:
        logger.info(
            f"Moving model to {device = } ...",
        )

    # Move the model to GPU if available
    model.to(
        device,  # type: ignore
    )

    if verbosity >= 1:
        logger.info(
            f"Moving model to {device = } DONE",
        )
        logger.info(
            f"{device = }",
        )
        log_model_info(
            model=model,
            model_name="model",
            logger=logger,
        )

    return model

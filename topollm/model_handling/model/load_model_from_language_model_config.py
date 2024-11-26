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

"""Load a model from a LanguageModelConfig object."""

import logging

import torch
from pydantic import BaseModel
from transformers import PreTrainedModel

from topollm.config_classes.language_model.language_model_config import LanguageModelConfig
from topollm.model_handling.model.get_model_class_from_task_type import get_model_class_from_task_type
from topollm.model_handling.model.load_model import load_model
from topollm.typing.enums import Verbosity

default_device = torch.device("cpu")
default_logger = logging.getLogger(__name__)


def load_model_from_language_model_config(
    language_model_config: LanguageModelConfig,
    from_pretrained_kwargs_instance: BaseModel | dict | None = None,
    device: torch.device = default_device,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> PreTrainedModel:
    """Load a model from a LanguageModelConfig object."""
    model_loading_class = get_model_class_from_task_type(
        task_type=language_model_config.task_type,
    )

    if from_pretrained_kwargs_instance is None:
        from_pretrained_kwargs_dict: dict = {}
    elif isinstance(
        from_pretrained_kwargs_instance,
        BaseModel,
    ):
        from_pretrained_kwargs_dict: dict = from_pretrained_kwargs_instance.model_dump()
    elif isinstance(
        from_pretrained_kwargs_instance,
        dict,
    ):
        from_pretrained_kwargs_dict: dict = from_pretrained_kwargs_instance
    else:
        msg: str = f"Unknown {from_pretrained_kwargs_instance = }"
        raise ValueError(
            msg,
        )

    # TODO: Add additional from_pretrained_kwargs entries here for the dropout parameters.

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "model_loading_class:\n%s",
            model_loading_class,
        )
        logger.info(
            "language_model_config:\n%s",
            language_model_config,
        )
        logger.info(
            "from_pretrained_kwargs:\n%s",
            from_pretrained_kwargs_dict,
        )

    model: PreTrainedModel = load_model(
        pretrained_model_name_or_path=language_model_config.pretrained_model_name_or_path,
        model_loading_class=model_loading_class,
        from_pretrained_kwargs=from_pretrained_kwargs_dict,
        device=device,
        verbosity=verbosity,
        logger=logger,
    )

    return model

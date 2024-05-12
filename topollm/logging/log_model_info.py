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

"""Logging utilities for model information."""

import logging
from typing import Any

from transformers import PreTrainedModel

default_logger = logging.getLogger(__name__)


def log_model_info(
    model: PreTrainedModel | Any,
    model_name: str = "model",
    logger: logging.Logger = default_logger,
) -> None:
    """Log model information."""
    logger.info(
        f"{type(model) = }",  # noqa: G004 - low overhead
    )
    logger.info(
        f"{model_name}:\n{model}",  # noqa: G004 - low overhead
    )

    if hasattr(
        model,
        "config",
    ):
        logger.info(
            f"{model_name}.config:\n{model.config}",  # noqa: G004 - low overhead
        )


def log_param_requires_grad_for_model(
    model: PreTrainedModel | Any,
    logger: logging.Logger = default_logger,
) -> None:
    """Log whether parameters require gradients for a model."""
    for name, param in model.named_parameters():
        logger.info(
            f"{name = }, {param.requires_grad = }",  # noqa: G004 - low overhead
        )

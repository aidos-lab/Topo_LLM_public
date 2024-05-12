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

"""Factory for gradient modifiers."""

import logging

import torch

from topollm.config_classes.finetuning.gradient_modifier.gradient_modifier_config import GradientModifierConfig
from topollm.model_finetuning.gradient_modifiers.gradient_modifier_do_nothing import GradientModifierDoNothing
from topollm.model_finetuning.gradient_modifiers.gradient_modifier_freeze_layers import GradientModifierFreezeLayers
from topollm.model_finetuning.model_modifiers.protocol import ModelModifier
from topollm.typing.enums import GradientModifierMode, Verbosity

default_logger = logging.getLogger(__name__)


def get_gradient_modifier(
    gradient_modifier_config: GradientModifierConfig,
    device: torch.device,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> ModelModifier:
    """Get a modifier for the given configuration."""
    mode: GradientModifierMode = gradient_modifier_config.mode

    if verbosity >= 1:
        logger.info(
            "gradient_modifier_config:\n%s",
            gradient_modifier_config,
        )
        logger.info(f"{mode = }")  # noqa: G004 - low overhead

    if mode == GradientModifierMode.DO_NOTHING:
        modifier = GradientModifierDoNothing(
            verbosity=verbosity,
            logger=logger,
        )
    elif mode == GradientModifierMode.FREEZE_LAYERS:
        if verbosity >= 1:
            logger.info("Freeze layers ...")

        modifier = GradientModifierFreezeLayers(
            target_modules_to_freeze=gradient_modifier_config.target_modules_to_freeze,
            verbosity=verbosity,
            logger=logger,
        )

        if verbosity >= 1:
            logger.info("Freeze layers DONE")
    else:
        msg = f"Unknown mode: {mode = }"
        raise ValueError(msg)

    return modifier

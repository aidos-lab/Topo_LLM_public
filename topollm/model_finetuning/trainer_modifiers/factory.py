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

from topollm.config_classes.finetuning.trainer_modifier.trainer_modifier_config import TrainerModifierConfig
from topollm.model_finetuning.trainer_modifiers.protocol import TrainerModifier
from topollm.model_finetuning.trainer_modifiers.trainer_modifier_do_nothing import TrainerModifierDoNothing
from topollm.typing.enums import TrainerModifierMode, Verbosity

default_logger = logging.getLogger(__name__)


def get_trainer_modifier(
    trainer_modifier_config: TrainerModifierConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> TrainerModifier:
    """Get a modifier for the given configuration."""
    mode: TrainerModifierMode = trainer_modifier_config.mode

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "trainer_modifier_config:\n%s",
            trainer_modifier_config,
        )
        logger.info(
            f"{mode = }",  # noqa: G004 - low overhead
        )

    match mode:
        case TrainerModifierMode.DO_NOTHING:
            if verbosity >= Verbosity.NORMAL:
                logger.info("Creating TrainerModifierDoNothing instance ...")

            modifier = TrainerModifierDoNothing(
                verbosity=verbosity,
                logger=logger,
            )
        # TODO: Implement the Callback version of the TrainerModifier
        case _:
            msg = f"Unknown mode: {mode = }"
            raise ValueError(msg)

    return modifier

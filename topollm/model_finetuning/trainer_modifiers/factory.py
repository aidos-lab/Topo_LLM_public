# Copyright 2024
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

"""Factory for gradient modifiers."""

import logging
from typing import TYPE_CHECKING

import datasets
import transformers

from topollm.config_classes.finetuning.finetuning_config import FinetuningConfig
from topollm.model_finetuning.trainer_modifiers.protocol import TrainerModifier
from topollm.model_finetuning.trainer_modifiers.trainer_modifier_do_nothing import TrainerModifierDoNothing
from topollm.model_finetuning.trainer_modifiers.trainer_modifier_wandb_prediction_progress_callback import (
    TrainerModifierWandbPredictionProgressCallback,
)
from topollm.typing.enums import TrainerModifierMode, Verbosity

if TYPE_CHECKING:
    from topollm.config_classes.finetuning.trainer_modifier.trainer_modifier_config import TrainerModifierConfig

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def get_trainer_modifier(
    finetuning_config: FinetuningConfig,
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast | None = None,
    dataset: datasets.Dataset | None = None,
    label_list: list[str] | None = None,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> TrainerModifier:
    """Get a modifier for the given configuration."""
    trainer_modifier_config: TrainerModifierConfig = finetuning_config.trainer_modifier
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
        case TrainerModifierMode.ADD_WANDB_PREDICTION_PROGRESS_CALLBACK:
            if verbosity >= Verbosity.NORMAL:
                logger.info("Creating TrainerModifierWandbPredictionProgressCallback instance ...")

            if tokenizer is None:
                msg = f"Tokenizer must be provided for {mode = }"
                raise ValueError(msg)
            if dataset is None:
                msg = f"Dataset must be provided for {mode = }"
                raise ValueError(msg)

            modifier = TrainerModifierWandbPredictionProgressCallback(
                finetuning_config=finetuning_config,
                tokenizer=tokenizer,
                dataset=dataset,
                label_list=label_list,
                verbosity=verbosity,
                logger=logger,
            )
        case _:
            msg = f"Unknown mode: {mode = }"
            raise ValueError(msg)

    return modifier

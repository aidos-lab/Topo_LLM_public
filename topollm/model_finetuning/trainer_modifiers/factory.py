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

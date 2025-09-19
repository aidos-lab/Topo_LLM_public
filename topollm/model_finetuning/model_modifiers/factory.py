"""Factory for model modifiers."""

import logging

import torch

from topollm.config_classes.finetuning.peft.peft_config import PEFTConfig
from topollm.config_classes.finetuning.peft.peft_config_to_lora_config import (
    peft_config_to_lora_config,
)
from topollm.model_finetuning.model_modifiers import model_modifier_lora, model_modifier_standard
from topollm.model_finetuning.model_modifiers.protocol import ModelModifier
from topollm.typing.enums import FinetuningMode, Verbosity

default_logger = logging.getLogger(__name__)


def get_model_modifier(
    peft_config: PEFTConfig,
    device: torch.device,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> ModelModifier:
    """Get a model modifier for the given configuration."""
    finetuning_mode = peft_config.finetuning_mode
    if verbosity >= Verbosity.NORMAL:
        logger.info(f"{finetuning_mode = }")  # noqa: G004 - low overhead

    if finetuning_mode == FinetuningMode.STANDARD:
        model_modifier = model_modifier_standard.ModelModifierStandard(
            logger=logger,
        )
    elif finetuning_mode == FinetuningMode.LORA:
        lora_config = peft_config_to_lora_config(
            peft_config=peft_config,
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info("Preparing LoRA adapter ...")
            logger.info(
                "lora_config:\n%s",
                lora_config,
            )

        model_modifier = model_modifier_lora.ModelModifierLora(
            lora_config=lora_config,
            device=device,
            verbosity=verbosity,
            logger=logger,
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info("Preparing LoRA adapter DONE.")
    else:
        msg = f"Unknown training mode: {finetuning_mode = }"
        raise ValueError(msg)

    return model_modifier

"""Factory for gradient modifiers."""

import logging

import torch

from topollm.config_classes.finetuning.gradient_modifier.gradient_modifier_config import GradientModifierConfig
from topollm.model_finetuning.gradient_modifiers.gradient_modifier_do_nothing import GradientModifierDoNothing
from topollm.model_finetuning.gradient_modifiers.gradient_modifier_freeze_layers import GradientModifierFreezeLayers
from topollm.model_finetuning.gradient_modifiers.protocol import GradientModifier
from topollm.typing.enums import GradientModifierMode, Verbosity

default_logger = logging.getLogger(__name__)


def get_gradient_modifier(
    gradient_modifier_config: GradientModifierConfig,
    device: torch.device,  # noqa: ARG001 - might be used in the future
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> GradientModifier:
    """Get a modifier for the given configuration."""
    mode: GradientModifierMode = gradient_modifier_config.mode

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "gradient_modifier_config:\n%s",
            gradient_modifier_config,
        )
        logger.info(f"{mode = }")  # noqa: G004 - low overhead

    if mode == GradientModifierMode.DO_NOTHING:
        if verbosity >= Verbosity.NORMAL:
            logger.info("Creating GradientModifierDoNothing instance ...")

        modifier = GradientModifierDoNothing(
            verbosity=verbosity,
            logger=logger,
        )
    elif mode == GradientModifierMode.FREEZE_LAYERS:
        if verbosity >= Verbosity.NORMAL:
            logger.info("Creating GradientModifierFreezeLayers instance ...")

        modifier = GradientModifierFreezeLayers(
            target_modules_to_freeze=gradient_modifier_config.target_modules_to_freeze,
            verbosity=verbosity,
            logger=logger,
        )
    else:
        msg = f"Unknown mode: {mode = }"
        raise ValueError(msg)

    return modifier

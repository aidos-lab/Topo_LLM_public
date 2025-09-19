"""Gradient modifier that does not modify the model."""

import logging

import peft.peft_model
from transformers import PreTrainedModel

from topollm.logging.log_model_info import log_param_requires_grad_for_model
from topollm.typing.enums import Verbosity

default_logger = logging.getLogger(__name__)


class GradientModifierDoNothing:
    """Gradient modifier that does not modify the model."""

    def __init__(
        self,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Initialize the model modifier."""
        self.verbosity = verbosity
        self.logger = logger

    def modify_gradients(
        self,
        model: PreTrainedModel | peft.peft_model.PeftModel,
    ) -> PreTrainedModel | peft.peft_model.PeftModel:
        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info("Using model without gradient modifications.")
            self.logger.info("Returning unmodified model.")

        if self.verbosity >= Verbosity.NORMAL:
            log_param_requires_grad_for_model(
                model=model,
                logger=self.logger,
            )

        return model

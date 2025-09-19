"""Model modifier that does not modify the model, i.e., leads to standard fine-tuning behavior."""

import logging

import peft.peft_model
from transformers import PreTrainedModel

from topollm.typing.enums import Verbosity

default_logger = logging.getLogger(__name__)


class ModelModifierStandard:
    """Model modifier that does not modify the model, i.e., leads to standard fine-tuning behavior."""

    def __init__(
        self,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Initialize the model modifier."""
        self.verbosity = verbosity
        self.logger = logger

    def modify_model(
        self,
        model: PreTrainedModel,
    ) -> peft.peft_model.PeftModel | PreTrainedModel:
        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info("Using base model without modifications.")
            self.logger.info("Returning unmodified model.")

        return model

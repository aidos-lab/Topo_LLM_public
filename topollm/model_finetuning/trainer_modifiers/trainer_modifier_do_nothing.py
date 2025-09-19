"""Gradient modifier that does not modify the model."""

import logging

import transformers

from topollm.typing.enums import Verbosity

default_logger = logging.getLogger(__name__)


class TrainerModifierDoNothing:
    """Gradient modifier that does not modify the model."""

    def __init__(
        self,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Initialize the model modifier."""
        self.verbosity = verbosity
        self.logger = logger

    def modify_trainer(
        self,
        trainer: transformers.Trainer,
    ) -> transformers.Trainer:
        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info("Returning unmodified trainer.")

        return trainer

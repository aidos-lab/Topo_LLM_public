import logging

import transformers

from topollm.typing.enums import Verbosity

default_logger = logging.getLogger(__name__)


def save_tuned_model(
    trainer: transformers.Trainer,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Save the tuned model to disk."""
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "Calling trainer.save_model() ...",
        )

    trainer.save_model()

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "Calling trainer.save_model() DONE",
        )

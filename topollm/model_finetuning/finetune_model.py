import logging

import transformers

from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def finetune_model(
    trainer: transformers.Trainer,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Finetune a model using the provided trainer."""
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Calling trainer.train() ...",
        )

    training_call_output = trainer.train(
        resume_from_checkpoint=False,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Calling trainer.train() DONE",
        )
        logger.info(
            "training_call_output:\n%s",
            training_call_output,
        )

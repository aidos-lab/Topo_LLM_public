# Copyright 2024
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_1 (author1@example.com)
# AUTHOR_2 (author2@example.com)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#


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

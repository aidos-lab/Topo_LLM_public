"""Evaluate the tuned model."""

import logging
import math
import pprint

import transformers

from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def evaluate_trainer(
    trainer: transformers.Trainer,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> dict:
    """Evaluate the model."""
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Evaluating the model ...",
        )

    eval_results: dict = trainer.evaluate()
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"eval_results:\n{pprint.pformat(object=eval_results)}",  # noqa: G004 - low overhead
        )

    # Since the model evaluation might not return the 'eval_loss' key, we need to check for it
    if "eval_loss" in eval_results:
        perplexity: float = math.exp(eval_results["eval_loss"])
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"perplexity:\n{perplexity:.2f}",  # noqa: G004 - low overhead
            )
    else:
        logger.warning(
            msg="Could not calculate perplexity, because 'eval_loss' was not in eval_results",
        )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Evaluating the model DONE",
        )

    return eval_results

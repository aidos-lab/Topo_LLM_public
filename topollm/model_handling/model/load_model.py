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


"""Load a model."""

import logging
import os

import torch
import transformers

from topollm.logging.log_model_info import log_model_info
from topollm.typing.enums import Verbosity

default_device = torch.device(device="cpu")
default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def load_model(
    pretrained_model_name_or_path: str | os.PathLike,
    model_loading_class: type = transformers.AutoModelForPreTraining,
    from_pretrained_kwargs: dict | None = None,
    device: torch.device = default_device,
    verbosity: int = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> transformers.PreTrainedModel:
    """Load the model based on the configuration.

    Args:
    ----
        pretrained_model_name_or_path:
            The name or path of the pretrained model.
        model_loading_class:
            The class to use for loading the model.
        from_pretrained_kwargs:
            The keyword arguments to pass to the from_pretrained() method.
        device:
            The device to move the model to.
        verbosity:
            The verbosity level.
        logger:
            The logger to use.

    """
    if from_pretrained_kwargs is None:
        from_pretrained_kwargs = {}

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"Loading model {pretrained_model_name_or_path = } ...",  # noqa: G004 - low overhead
        )
        logger.info(
            "from_pretrained_kwargs:\n%s",
            from_pretrained_kwargs,
        )

    if not hasattr(
        model_loading_class,
        "from_pretrained",
    ):
        msg = f"Does not have a .from_pretrained() method: {model_loading_class = }"
        raise ValueError(msg)

    model: transformers.PreTrainedModel = model_loading_class.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        **from_pretrained_kwargs,
    )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"Loading model {pretrained_model_name_or_path = } DONE",  # noqa: G004 - low overhead
        )

    if not isinstance(
        model,
        transformers.PreTrainedModel,
    ):
        msg = f"model is not of type PreTrainedModel: {type(model) = }"
        raise TypeError(msg)

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"Moving model to {device = } ...",  # noqa: G004 - low overhead
        )

    # Move the model to GPU if available
    model.to(
        device,  # type: ignore - torch.device
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"Moving model to {device = } DONE",  # noqa: G004 - low overhead
        )
        logger.info(
            "device:\n%s",
            device,
        )
        log_model_info(
            model=model,
            model_name="model",
            logger=logger,
        )

    return model

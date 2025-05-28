# Copyright 2024-2025
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


"""Load a model from a LanguageModelConfig object."""

import logging
import pprint

import torch
from pydantic import BaseModel
from transformers import PreTrainedModel

from topollm.config_classes.language_model.language_model_config import LanguageModelConfig
from topollm.model_handling.model.get_from_pretrained_kwargs_dict_for_dropout_parameters import (
    get_from_pretrained_kwargs_dict_for_dropout_parameters,
)
from topollm.model_handling.model.get_model_class_from_task_type import get_model_class_from_task_type
from topollm.model_handling.model.load_model import load_model
from topollm.typing.enums import Verbosity

default_device = torch.device(
    device="cpu",
)
default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def load_model_from_language_model_config(
    language_model_config: LanguageModelConfig,
    from_pretrained_kwargs_instance: BaseModel | dict | None = None,
    device: torch.device = default_device,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> PreTrainedModel:
    """Load a model from a LanguageModelConfig object."""
    model_loading_class = get_model_class_from_task_type(
        task_type=language_model_config.task_type,
    )

    # # # #
    # Prepare from_pretrained_kwargs_dict from the function input from_pretrained_kwargs_instance.
    if from_pretrained_kwargs_instance is None:
        from_pretrained_kwargs_from_function_input_dict: dict = {}
    elif isinstance(
        from_pretrained_kwargs_instance,
        BaseModel,
    ):
        from_pretrained_kwargs_from_function_input_dict: dict = from_pretrained_kwargs_instance.model_dump()
    elif isinstance(
        from_pretrained_kwargs_instance,
        dict,
    ):
        from_pretrained_kwargs_from_function_input_dict: dict = from_pretrained_kwargs_instance
    else:
        msg: str = f"Unknown {from_pretrained_kwargs_instance = }"
        raise ValueError(
            msg,
        )

    # # # #
    # Add the keyword arguments for the dropout parameters to the from_pretrained_kwargs_dict.
    from_pretrained_kwargs_dict_for_dropout_parameters: dict = get_from_pretrained_kwargs_dict_for_dropout_parameters(
        language_model_config=language_model_config,
    )

    from_pretrained_kwargs_dict: dict = {
        **from_pretrained_kwargs_from_function_input_dict,
        **from_pretrained_kwargs_dict_for_dropout_parameters,
    }

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "model_loading_class:\n%s",
            model_loading_class,
        )
        logger.info(
            "language_model_config:\n%s",
            pprint.pformat(object=language_model_config),
        )
        logger.info(
            "from_pretrained_kwargs:\n%s",
            pprint.pformat(object=from_pretrained_kwargs_dict),
        )

    model: PreTrainedModel = load_model(
        pretrained_model_name_or_path=language_model_config.pretrained_model_name_or_path,
        model_loading_class=model_loading_class,
        from_pretrained_kwargs=from_pretrained_kwargs_dict,
        device=device,
        verbosity=verbosity,
        logger=logger,
    )

    return model

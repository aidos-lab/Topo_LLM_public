"""Logging utilities for model information."""

import logging
from typing import Any

from transformers import PreTrainedModel

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)

default_logger_block_separator: str = "=" * 80
default_logger_section_separator: str = "-" * 80


def log_model_info(
    model: PreTrainedModel | Any,  # noqa: ANN401 - fixing typing issue with the transformers models
    model_name: str = "model",
    logger_section_separator: str | None = default_logger_section_separator,
    logger_block_separator: str | None = default_logger_block_separator,
    logger: logging.Logger = default_logger,
) -> None:
    """Log model information."""
    if logger_block_separator is not None:
        logger.info(
            msg=logger_block_separator,
        )

    logger.info(
        msg=f"{type(model) = }",  # noqa: G004 - low overhead
    )
    logger.info(
        msg=f"{model_name}:\n{model}",  # noqa: G004 - low overhead
    )

    if logger_section_separator is not None:
        logger.info(
            msg=logger_section_separator,
        )

    if hasattr(
        model,
        "config",
    ):
        logger.info(
            msg=f"{model_name}.config:\n{model.config}",  # noqa: G004 - low overhead
        )

    if logger_section_separator is not None:
        logger.info(
            msg=logger_section_separator,
        )

    log_named_parameters_and_state_dict_for_model(
        model=model,
        logger_section_separator=logger_section_separator,
        logger=logger,
    )

    if logger_section_separator is not None:
        logger.info(
            msg=logger_section_separator,
        )

    log_shapes_of_parameters(
        model=model,
        logger=logger,
    )

    if logger_section_separator is not None:
        logger.info(
            msg=logger_section_separator,
        )

    log_param_requires_grad_for_model(
        model=model,
        logger=logger,
    )

    if logger_block_separator is not None:
        logger.info(
            msg=logger_block_separator,
        )


def log_named_parameters_and_state_dict_for_model(
    model: PreTrainedModel | Any,  # noqa: ANN401 - fixing typing issue with the transformers models
    logger_section_separator: str | None = default_logger_section_separator,
    logger: logging.Logger = default_logger,
) -> None:
    """Log named parameters and state dict for a model."""
    if logger_section_separator is not None:
        logger.info(
            msg=logger_section_separator,
        )

    # Log the names of the named parameters
    logger.info(
        msg=f"{model.__class__.__name__} model.named_parameters():",  # noqa: G004 - low overhead
    )
    names_list = [name for name, _ in model.named_parameters()]
    logger.info(
        msg=f"names_list:\n{names_list}",  # noqa: G004 - low overhead
    )

    if logger_section_separator is not None:
        logger.info(
            msg=logger_section_separator,
        )

    # Log the keys in the state dict
    logger.info(
        msg=f"{model.__class__.__name__} model.state_dict().keys():",  # noqa: G004 - low overhead
    )
    state_dict_keys_list = list(model.state_dict().keys())
    logger.info(
        msg=f"state_dict_keys_list:\n{state_dict_keys_list}",  # noqa: G004 - low overhead
    )
    if logger_section_separator is not None:
        logger.info(
            msg=logger_section_separator,
        )


def log_shapes_of_parameters(
    model: PreTrainedModel | Any,  # noqa: ANN401 - fixing typing issue with the transformers models
    logger: logging.Logger = default_logger,
) -> None:
    """Log the shapes of the parameters in the state dict for a model."""
    for key, value in model.state_dict().items():
        if hasattr(
            value,
            "shape",
        ):
            logger.info(
                msg=f"{key = }; {value.shape = }.",  # noqa: G004 - low overhead
            )
        else:
            logger.info(
                msg=f"{key = } has no shape attribute.",  # noqa: G004 - low overhead
            )


def log_param_requires_grad_for_model(
    model: PreTrainedModel | Any,  # noqa: ANN401 - fixing typing issue with the transformers models
    logger: logging.Logger = default_logger,
) -> None:
    """Log whether parameters require gradients for a model."""
    for name, param in model.named_parameters():
        logger.info(
            msg=f"{name = }, {param.requires_grad = }",  # noqa: G004 - low overhead
        )

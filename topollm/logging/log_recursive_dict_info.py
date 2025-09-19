import logging

import torch

from topollm.logging.log_array_info import log_tensor_info
from topollm.logging.log_list_info import log_list_info

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def log_recursive_dict_info(
    dictionary: dict,
    dictionary_name: str,
    logger: logging.Logger = default_logger,
) -> None:
    """Log information about the dictionary."""
    logger.info(
        msg=f"type({dictionary_name}):\n{type(dictionary)}",  # noqa: G004 - low overhead
    )

    logger.info(
        msg=f"{dictionary_name}.keys():\n{dictionary.keys()}",  # noqa: G004 - low overhead
    )

    # Log the types of the values in the dictionary
    logger.info(
        msg="Logging types of values in the dictionary.",
    )
    for key, value in dictionary.items():
        logger.info(
            msg=f"{key = }; {type(value) = }.",  # noqa: G004 - low overhead
        )

    # Depending on the type of the values, log additional information
    logger.info(
        msg=f"Recursively logging the values in the dictionary {dictionary_name = }.",  # noqa: G004 - low overhead
    )
    for key, value in dictionary.items():
        if isinstance(
            value,
            dict,
        ):
            log_recursive_dict_info(
                dictionary=value,
                dictionary_name=f"{dictionary_name}[{key}]",
                logger=logger,
            )
        elif isinstance(
            value,
            list,
        ):
            log_list_info(
                list_=value,
                list_name=f"{dictionary_name}[{key}]",
                logger=logger,
            )
        elif isinstance(
            value,
            torch.Tensor,
        ):
            log_tensor_info(
                tensor=value,
                tensor_name=f"{dictionary_name}[{key}]",
                logger=logger,
            )
        else:
            logger.info(
                msg=f"No logging implemented for this type: {dictionary_name}[{key}]; {type(value) = }. Skipping.",  # noqa: G004 - low overhead
            )

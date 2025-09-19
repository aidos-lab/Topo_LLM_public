"""Truncate the length of the description string if it is too long."""

import logging

from topollm.config_classes.constants import FILE_NAME_TRUNCATION_LENGTH

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def truncate_length_of_desc(
    desc: str,
    truncation_length: int = FILE_NAME_TRUNCATION_LENGTH,
    logger: logging.Logger = default_logger,
) -> str:
    """Truncate the length of the description string if it is too long."""
    if len(desc) > truncation_length:
        logger.warning(
            msg=f"Too long:\n{desc = }\n{len(desc) = }\nWill be truncated to {truncation_length = } characters.",  # noqa: G004 - low overhead
        )
        desc = desc[:truncation_length]

    return desc

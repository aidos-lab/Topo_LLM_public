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


"""Log information about a list."""

import logging
import pprint

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def log_list_info(
    list_: list,
    list_name: str,
    max_log_elements: int = 20,
    logger: logging.Logger = default_logger,
) -> None:
    """Log information about a list.

    Args:
        list_ (list):
            The list to log information about.
        list_name (str):
            The name of the list.
        max_log_elements (int, optional):
            The maximum number of elements to log for the head and tail of the list.
            Defaults to 20.
        logger (logging.Logger, optional):
            The logger to log information to.

    Returns:
        None

    Side effects:
        Logs information about the list to the logger.

    """
    logger.info(
        msg=f"len({list_name}):\n{len(list_)}",  # noqa: G004 - low overhead
    )
    logger.info(
        msg=f"{list_name}[:{max_log_elements}]:\n{pprint.pformat(list_[:max_log_elements])}",  # noqa: G004 - low overhead
    )
    logger.info(
        msg=f"{list_name}[-{max_log_elements}:]:\n{pprint.pformat(list_[-max_log_elements:])}",  # noqa: G004 - low overhead
    )

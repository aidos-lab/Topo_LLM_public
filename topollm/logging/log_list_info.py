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

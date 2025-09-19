"""Create and configure a global logger."""

import logging
import pathlib

from topollm.logging.setup_exception_logging import setup_exception_logging


def create_and_configure_global_logger(
    name: str = __name__,
    file: str = __file__,
) -> logging.Logger:
    """Create and configure a global logger."""
    global_logger: logging.Logger = logging.getLogger(
        name=name,
    )
    global_logger.setLevel(
        level=logging.INFO,
    )
    logging_formatter = logging.Formatter(
        fmt="[%(asctime)s][%(levelname)8s][%(name)s] %(message)s (%(filename)s:%(lineno)s)",
    )

    logging_file_path = pathlib.Path(
        pathlib.Path(file).parent,
        "logs",
        f"{pathlib.Path(file).stem}.log",
    )
    pathlib.Path.mkdir(
        self=logging_file_path.parent,
        parents=True,
        exist_ok=True,
    )

    logging_file_handler = logging.FileHandler(
        filename=logging_file_path,
    )
    logging_file_handler.setFormatter(
        fmt=logging_formatter,
    )
    global_logger.addHandler(
        hdlr=logging_file_handler,
    )

    logging_console_handler = logging.StreamHandler()
    logging_console_handler.setFormatter(
        fmt=logging_formatter,
    )
    global_logger.addHandler(
        hdlr=logging_console_handler,
    )

    setup_exception_logging(
        logger=global_logger,
    )

    return global_logger

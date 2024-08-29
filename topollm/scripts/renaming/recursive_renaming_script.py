# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (ruppik@hhu.de)
# Julius von Rohrscheidt (julius.rohrscheidt@helmholtz-muenchen.de)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Recursively rename all directories within a base directory that contain a specific substring."""

import logging
import os
import pathlib
import sys

from topollm.config_classes.constants import TOPO_LLM_REPOSITORY_BASE_PATH
from topollm.typing.enums import Verbosity

default_logger = logging.getLogger(__name__)


def configure_logger() -> logging.Logger:
    """Configure the default logger to print to standard output."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all log levels.

    # Create a stream handler for logging to stdout.
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)  # Adjust this level as needed.

    # Create a formatter and set it for the handler.
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    # Add the handler to the logger.
    if not logger.handlers:  # Avoid adding multiple handlers if already configured.
        logger.addHandler(handler)

    return logger


def recursively_rename_in_directory_names(
    base_dir: os.PathLike,
    old_substring: str,
    new_substring: str,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> tuple[
    int,
    int,
]:
    """Recursively rename all directories within the given base directory that contain a specific substring.

    Args:
    ----
        base_dir:
            The path to the base directory where the renaming process will start.
        old_substring:
            The substring to search for in directory names.
        new_substring:
            The substring to replace the old one with.
        verbosity:
            The verbosity level for logging.
        logger:
            The logger to use for logging.

    Returns:
    -------
        A tuple containing:
        - The total number of directories that were renamed.
        - The total number of errors encountered during the renaming process.

    """
    rename_count = 0
    error_count = 0

    # Check if the base directory exists
    base_dir = pathlib.Path(base_dir)

    if not base_dir.exists() or not base_dir.is_dir():
        logger.exception(
            f"Error: The base directory '{base_dir = }' does not exist or is not a directory.",  # noqa: G004 - low overhead
        )
        logger.info("Exiting the renaming process.")
        sys.exit(1)

    # Walk through the directory tree from the base directory.
    for root, dirs, _ in os.walk(
        top=base_dir,
        topdown=False,  # Avoid renaming parent directories before their children.
    ):
        for dir_name in dirs:
            if old_substring in dir_name:
                # Create the new directory name by replacing the old substring with the new one.
                new_name = dir_name.replace(
                    old_substring,
                    new_substring,
                )

                old_path = pathlib.Path(
                    root,
                    dir_name,
                )
                new_path = pathlib.Path(
                    root,
                    new_name,
                )

                try:
                    if verbosity >= Verbosity.VERBOSE:
                        logger.info(
                            f"Renaming {old_path = } to {new_path = }.",  # noqa: G004 - low overhead
                        )

                    # Rename the directory.
                    pathlib.Path.rename(
                        old_path,
                        new_path,
                    )
                    rename_count += 1
                except TypeError as e:
                    # Log the error and continue.
                    logger.exception(
                        f"Error renaming {old_path} to {new_path}: {e}",  # noqa: G004 - low overhead
                    )
                    error_count += 1

    return rename_count, error_count


def main() -> None:
    """Call the recursive renaming script."""
    logger = configure_logger()
    verbosity = Verbosity.VERBOSE

    # # # #
    # Example configuration for the base directory and the substrings to replace.
    #
    # base_dir = pathlib.Path(
    #     TOPO_LLM_REPOSITORY_BASE_PATH,
    #     "data/analysis/prepared",
    # )

    base_dir = pathlib.Path(
        TOPO_LLM_REPOSITORY_BASE_PATH,
        "data/analysis/twonn",
    )

    # old_substring = "<AggregationType.MEAN: 'mean'>"
    # new_substring = "mean"

    old_substring = "task-MASKED_LM"
    new_substring = "task-masked_lm"

    # Call the renaming function.
    rename_count, error_count = recursively_rename_in_directory_names(
        base_dir=base_dir,
        old_substring=old_substring,
        new_substring=new_substring,
        verbosity=verbosity,
        logger=logger,
    )

    logger.info(
        f"Renaming completed. {rename_count = } directories were renamed.",  # noqa: G004 - low overhead
    )
    if error_count > 0:
        logger.warning(
            f"{error_count = } errors were encountered during the renaming process.",  # noqa: G004 - low overhead
        )


if __name__ == "__main__":
    main()

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

import argparse
import logging
import os
import pathlib
import sys
from dataclasses import dataclass

from tqdm import tqdm

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


@dataclass
class RenameOperation:
    """Class representing a single renaming operation."""

    base_dir: pathlib.Path
    old_substring: str
    new_substring: str


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

    logger.info(
        f"Renaming completed. {rename_count = } directories were renamed.",  # noqa: G004 - low overhead
    )
    if error_count > 0:
        logger.warning(
            f"{error_count = } errors were encountered during the renaming process.",  # noqa: G004 - low overhead
        )

    return rename_count, error_count


def batch_rename_operations(
    operations: list[RenameOperation],
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Perform multiple renaming operations as specified in the operations list.

    Args:
    ----
        operations: A list of RenameOperation instances.
        verbosity: The verbosity level for logging.
        logger: The logger to use for logging.

    """
    for operation in tqdm(
        operations,
        desc="Renaming operations",
    ):
        rename_count, error_count = recursively_rename_in_directory_names(
            base_dir=operation.base_dir,
            old_substring=operation.old_substring,
            new_substring=operation.new_substring,
            verbosity=verbosity,
            logger=logger,
        )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Recursively rename directories containing a specific substring.",
    )

    subparsers = parser.add_subparsers(
        dest="mode",
        required=True,
        help="Mode of operation",
    )

    # Single operation mode
    single_parser = subparsers.add_parser(
        "single",
        help="Perform a single renaming operation",
    )
    single_parser.add_argument(
        "--base-dir",
        type=pathlib.Path,
        required=True,
        help="Base directory where the renaming process will start.",
    )
    single_parser.add_argument(
        "--old-substring",
        type=str,
        required=True,
        help="The substring to search for in directory names.",
    )
    single_parser.add_argument(
        "--new-substring",
        type=str,
        required=True,
        help="The substring to replace the old one with.",
    )

    # Batch operation mode
    batch_parser = subparsers.add_parser(  # noqa: F841 - we might want to use this in the future
        "batch",
        help="Perform multiple renaming operations preconfigured in the python script.",
    )

    return parser.parse_args()


def main() -> None:
    """Handle command-line arguments and call the appropriate renaming function."""
    args = parse_arguments()

    logger = configure_logger()
    verbosity = Verbosity.VERBOSE

    if args.mode == "single":
        # Call the renaming function.
        rename_count, error_count = recursively_rename_in_directory_names(
            base_dir=args.base_dir,
            old_substring=args.old_substring,
            new_substring=args.new_substring,
            verbosity=verbosity,
            logger=logger,
        )
    elif args.mode == "batch":
        # Batch operation mode
        operations = [
            RenameOperation(
                base_dir=pathlib.Path(
                    TOPO_LLM_REPOSITORY_BASE_PATH,
                    "data/analysis/twonn",
                ),
                old_substring="task-MASKED_LM",
                new_substring="task-masked_lm",
            ),
            RenameOperation(
                base_dir=pathlib.Path(
                    TOPO_LLM_REPOSITORY_BASE_PATH,
                    "data/analysis/prepared",
                ),
                old_substring="<AggregationType.MEAN: 'mean'>",
                new_substring="mean",
            ),
            RenameOperation(
                base_dir=pathlib.Path(
                    TOPO_LLM_REPOSITORY_BASE_PATH,
                    "data/embeddings/perplexity",
                ),
                old_substring="task-MASKED_LM",
                new_substring="task-masked_lm",
            ),
            # Add more RenameOperation instances as needed
        ]
        batch_rename_operations(
            operations=operations,
            verbosity=verbosity,
            logger=logger,
        )

    logger.info("Exiting the renaming script.")


if __name__ == "__main__":
    main()

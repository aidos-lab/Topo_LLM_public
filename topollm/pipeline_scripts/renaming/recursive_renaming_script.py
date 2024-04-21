# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Julius von Rohrscheidt (julius.rohrscheidt@helmholtz-muenchen.de)
# Benjamin Ruppik (ruppik@hhu.de)
#
# This code was generated with the help of AI writing assistants
# including GitHub Copilot, ChatGPT, Bing Chat.
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

logger = logging.getLogger(__name__)


def rename_in_directory(
    base_dir: os.PathLike,
    old_substring: str,
    new_substring: str,
    logger: logging.Logger = logger,
) -> tuple[int, int]:
    """Recursively rename all directories within the given base directory that contain a specific substring.

    Args:
    ----
        base_dir: The path to the base directory where the renaming process will start.
        old_substring: The substring to search for in directory names.
        new_substring: The substring to replace the old one with.

    Returns:
    -------
        A tuple containing:
        - The total number of directories that were renamed.
        - The total number of errors encountered during the renaming process.

    """
    rename_count = 0
    error_count = 0

    # Walk through the directory tree from the base directory.
    for root, dirs, _ in os.walk(
        top=base_dir,
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
                    # Rename the directory.
                    pathlib.Path.rename(
                        old_path,
                        new_path,
                    )
                    rename_count += 1
                except TypeError as e:
                    # Log the error and continue.
                    print(f"Error renaming {old_path} to {new_path}: {e}")
                    error_count += 1

    return rename_count, error_count


def main() -> None:
    """Call the recursive renaming script."""
    # Example configuration for the base directory and the substrings to replace.
    base_dir = pathlib.Path(
        # "/home/benjamin_ruppik/git-source/Topo_LLM/data/embeddings/metadata",
        "/home/benjamin_ruppik/git-source/Topo_LLM/data/analysis/prepared",
    )
    old_substring = "<AggregationType.MEAN: 'mean'>"
    new_substring = "mean"

    # Call the renaming function.
    rename_count, error_count = rename_in_directory(
        base_dir=base_dir,
        old_substring=old_substring,
        new_substring=new_substring,
    )

    print(f"Renaming completed. {rename_count = } directories were renamed.")
    if error_count > 0:
        print(f"{error_count = } errors were encountered during the renaming process.")


if __name__ == "__main__":
    main()

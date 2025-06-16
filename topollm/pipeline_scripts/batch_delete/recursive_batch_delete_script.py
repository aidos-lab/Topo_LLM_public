# Copyright 2024
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_2 (author2@example.com)
# AUTHOR_1 (author1@example.com)
#
# This code was generated with the help of AI writing assistants
# including GitHub Copilot, ChatGPT, Bing Chat.
#


"""Recursively go through directories starting from a root and delete all filtes matching a certain pattern."""

import glob
import logging
import os
import pathlib
import sys

default_logger = logging.getLogger(__name__)
default_logger.setLevel(logging.INFO)
default_logger.addHandler(
    logging.StreamHandler(
        stream=sys.stdout,
    ),
)


def delete_files_except(
    root_directory: os.PathLike,
    delete_pattern: str = "embeddings_token_lvl_*[.pkl|.npy]",
    retain_prefix_pattern: str = "embeddings_token_lvl_50000",
    *,
    dry_run: bool = True,
    logger: logging.Logger = default_logger,
) -> list[pathlib.Path] | None:
    """Delete files that match the pattern except for those starting with the given retain pattern.

    :param root_directory: The root directory to start traversing.
    :param delete_pattern: The pattern to match files to be deleted (e.g., "embeddings_token_lvl_*").
    :param retain_prefix_pattern: The prefix pattern to retain (e.g., "embeddings_token_lvl_50000").
    :param dry_run: If True, lists files to be deleted without actually deleting them. Defaults to False.

    :return: A list of files to be deleted if dry_run is True, else None.
    """
    logger.info(f"{delete_pattern = }")  # noqa: G004 - low overhead
    files_to_delete: list[pathlib.Path] = []  # List to store files to be deleted

    # Traverse through the entire folder structure recursively
    for current_dir, _, _ in os.walk(
        root_directory,
    ):
        logger.info(f"{current_dir = }")  # noqa: G004 - low overhead

        # Find all files with the specified pattern
        all_files = glob.glob(  # noqa: PTH207 - we tested the function with glob.glob
            pathname=delete_pattern,
            root_dir=current_dir,
        )

        logger.info(f"{len(all_files) = }")  # noqa: G004 - low overhead

        # Add files that do not start with the retain pattern to the list
        for file in all_files:
            if pathlib.Path(file).name.startswith(
                retain_prefix_pattern,
            ):
                logger.info(f"retained: {file = }")  # noqa: G004 - low overhead
                continue

            absolute_file_to_delete_path = pathlib.Path(
                current_dir,
                file,
            )
            files_to_delete.append(
                absolute_file_to_delete_path,
            )

    logger.info(f"{len(files_to_delete) = }")  # noqa: G004 - low overhead

    if dry_run:
        # If it is a dry run, just return the list of files to be deleted
        return files_to_delete

    # Otherwise, delete the files
    for file in files_to_delete:
        pathlib.Path(
            file,
        ).unlink()
        logger.info(f"deleted: {file = }")  # noqa: G004 - low overhead

    # Return None if files are deleted
    return None


def main() -> None:
    """Call the recursive deletion script."""
    # Configuration for the base directory and the matching names.
    base_dir = pathlib.Path(
        "$HOME/git-source/Topo_LLM/data/analysis/prepared",
    )

    delete_pattern = "embeddings_token_lvl_*[.pkl|.npy]"
    retain_prefix_pattern = "embeddings_token_lvl_50000"

    # Call the deletion function.
    files_to_delete = delete_files_except(
        root_directory=base_dir,
        delete_pattern=delete_pattern,
        retain_prefix_pattern=retain_prefix_pattern,
        dry_run=False,
    )

    if files_to_delete:
        default_logger.info("Files to be deleted (dry run):")
        for file in files_to_delete:
            default_logger.info(file)
        default_logger.info(f"{len(files_to_delete) = } files to be deleted.")  # noqa: G004 - low overhead
        default_logger.info("Run the script with `dry_run=False` to delete the files.")


if __name__ == "__main__":
    main()

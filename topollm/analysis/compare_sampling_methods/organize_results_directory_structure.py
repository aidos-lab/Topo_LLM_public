# Copyright 2024
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


"""Tools for organizing the directory structure of the results of the analysis."""

import logging
import pathlib

from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def build_results_directory_structure(
    analysis_base_directory: pathlib.Path,
    data_dir: pathlib.Path,
    analysis_output_subdirectory_partial_relative_path: pathlib.Path = pathlib.Path(
        "sample_sizes",
    ),
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> pathlib.Path:
    """Build the results directory structure where we can save the results of the analysis."""
    results_base_directory = pathlib.Path(
        data_dir,
        "analysis",
        analysis_output_subdirectory_partial_relative_path,
        analysis_base_directory.relative_to(
            data_dir,
        ),
    )
    results_base_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{results_base_directory = }",  # noqa: G004 - low overhead
        )

    return results_base_directory

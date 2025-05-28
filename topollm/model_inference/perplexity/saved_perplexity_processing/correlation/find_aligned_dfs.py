# Copyright 2024
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_1 (author1@example.com)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#


import logging
import os
import pathlib

from tqdm import tqdm

from topollm.model_inference.perplexity.saved_perplexity_processing.correlation.aligned_df_containers import (
    AlignedDF,
    AlignedDFCollection,
)
from topollm.typing.enums import Verbosity

default_logger = logging.getLogger(__name__)


def find_aligned_dfs(
    root_dir: os.PathLike,
    dataset: str | None = None,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> AlignedDFCollection:
    """Recursively find all aligned_df.csv files in the given directory that match the dataset pattern.

    Args:
    ----
        root_dir:
            Root directory to start the search.
        dataset:
            The dataset identifier to filter the models.
            None means no filtering.
        verbosity:
            The verbosity level for logging.
        logger:
            The logger instance to use for logging.

    Returns:
    -------
        AlignedDFCollection:
            A collection of AlignedDF objects.

    """
    aligned_df_collection = AlignedDFCollection()

    for dirpath, _, filenames in tqdm(
        os.walk(
            root_dir,
        ),
    ):
        if "aligned_df.csv" in filenames:
            file_path = pathlib.Path(
                dirpath,
                "aligned_df.csv",
            )

            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    f"Found aligned_df.csv file: {file_path = }",  # noqa: G004 - low overhead
                )

            aligned_df_object = AlignedDF(
                file_path=file_path,
            )

            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    f"{aligned_df_object.metadata = }",  # noqa: G004 - low overhead
                )

            if aligned_df_object.metadata.dataset == dataset or dataset is None:
                aligned_df_collection.add_aligned_df(
                    aligned_df=aligned_df_object,
                )

    return aligned_df_collection

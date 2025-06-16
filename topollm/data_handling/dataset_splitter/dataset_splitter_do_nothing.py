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


"""Do nothing with the dataset dict."""

import logging

import datasets

from topollm.typing.enums import Verbosity

logger: logging.Logger = logging.getLogger(
    name=__name__,
)


class DatasetSplitterDoNothing:
    """Do nothing with the dataset dict."""

    def __init__(
        self,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = logger,
    ) -> None:
        """Initialize the dataset splitter."""
        self.verbosity: Verbosity = verbosity
        self.logger: logging.Logger = logger

    def split_dataset(
        self,
        dataset_dict: datasets.DatasetDict,
    ) -> datasets.DatasetDict:
        """Return the dataset_dict unchanged."""
        if self.verbosity >= 1:
            self.logger.info("Returning unchanged dataset_dict.")
        return dataset_dict

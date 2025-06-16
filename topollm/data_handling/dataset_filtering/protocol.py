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


"""Protocol for filtering a dataset dict."""

from typing import Protocol

import datasets


class DatasetFilter(Protocol):
    """Protocol for filtering a dataset."""

    def filter_dataset_dict(
        self,
        dataset_dict: datasets.DatasetDict,
    ) -> datasets.DatasetDict:
        """Return a filtered version of the dataset."""
        ...  # pragma: no cover

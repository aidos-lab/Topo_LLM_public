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


"""Protocol for splitting a dataset."""

from typing import Protocol

import datasets


class DatasetSubsampler(Protocol):
    """Protocol for subsampling from a dataset."""

    def subsample_dataset(
        self,
        dataset: datasets.Dataset,
    ) -> datasets.Dataset:
        """Subsample the dataset."""
        ...  # pragma: no cover

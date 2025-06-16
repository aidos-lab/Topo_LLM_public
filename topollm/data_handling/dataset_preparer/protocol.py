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


"""Protocol for preparing a dataset."""

from typing import Protocol

import datasets


class DatasetPreparer(Protocol):
    """Protocol for preparing a dataset."""

    def prepare_dataset(
        self,
    ) -> datasets.Dataset:
        """Load and prepare a dataset."""
        ...  # pragma: no cover

    def __len__(
        self,
    ) -> int:
        """Return the number of samples in the dataset."""
        ...  # pragma: no cover

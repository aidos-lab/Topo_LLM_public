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


"""Protocol class for sampling in the embeddings data preparation."""

from typing import Protocol

import numpy as np

from topollm.embeddings_data_prep.prepared_data_containers import PreparedData


class SubsetSampler(Protocol):
    """Protocol class for sampling in the embeddings data preparation."""

    def sample_subsets(
        self,
        input_data: PreparedData,
    ) -> tuple[
        PreparedData,
        np.ndarray,
    ]:
        """Sample subsets of the arrays and metadata in the input_data.

        Returns
        -------
            A tuple of the prepared data and the sampling indices.

        """
        ...  # pragma: no cover

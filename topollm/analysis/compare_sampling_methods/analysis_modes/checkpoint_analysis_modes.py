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


"""Configuration for different modes of checkpoint analysis."""

from dataclasses import dataclass, field
from itertools import product

import pandas as pd

from topollm.analysis.compare_sampling_methods.analysis_modes.core_combinations import (
    BaseAnalysisCombination,
    BaseAnalysisModes,
)


@dataclass
class CheckpointAnalysisCombination(BaseAnalysisCombination):
    """Extends the core combination with model_partial_name and seed."""

    model_partial_name: str
    language_model_seed: int


@dataclass
class CheckpointAnalysisModes(BaseAnalysisModes[CheckpointAnalysisCombination]):
    """Extends the core modes with model_partial_name and seed lists.

    Subclass the generic BaseAnalysisModes with C=CheckpointAnalysisCombination.
    """

    model_partial_name_list: list[str] = field(
        default_factory=list,
    )
    language_model_seed_list: list[int] = field(
        default_factory=list,
    )

    def from_concatenated_df(
        self,
        concatenated_df: pd.DataFrame,
    ) -> None:
        """Populate all inherited fields plus model_partial_name and language_model_seed."""
        # First populate the inherited fields
        super().from_concatenated_df(
            concatenated_df=concatenated_df,
        )

        # Then populate the new fields
        self.model_partial_name_list = concatenated_df["model_partial_name"].unique().tolist()
        self.language_model_seed_list = concatenated_df["model_seed"].unique().tolist()

    def all_combinations(
        self,
    ) -> list[CheckpointAnalysisCombination]:
        """Return all possible combinations, now including model_partial_name and language_model_seed.

        Return a list of CheckpointAnalysisCombination (C=CheckpointAnalysisCombination).
        This satisfies the base signature: 'List[C]' where C is CheckpointAnalysisCombination.
        """
        # First get the base lists from BaseAnalysisModes
        data_full_list: list[str] = self.data_full_list
        data_subsampling_split_list: list[str] = self.data_subsampling_split_list
        data_subsampling_sampling_mode_list: list[str] = self.data_subsampling_sampling_mode_list
        embedding_data_handler_mode_list: list[str] = self.embedding_data_handler_mode_list

        # Then add the two new lists
        model_partial_name_list: list[str] = self.model_partial_name_list
        language_model_seed_list: list[int] = self.language_model_seed_list

        return [
            CheckpointAnalysisCombination(*combo)
            for combo in product(
                data_full_list,
                data_subsampling_split_list,
                data_subsampling_sampling_mode_list,
                embedding_data_handler_mode_list,
                model_partial_name_list,
                language_model_seed_list,
            )
        ]

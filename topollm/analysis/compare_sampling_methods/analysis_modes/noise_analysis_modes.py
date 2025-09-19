"""Configuration for different modes of checkpoint analysis."""

from dataclasses import dataclass, field
from itertools import product

import pandas as pd

from topollm.analysis.compare_sampling_methods.analysis_modes.core_combinations import (
    BaseAnalysisCombination,
    BaseAnalysisModes,
)


@dataclass
class NoiseAnalysisCombination(BaseAnalysisCombination):
    """Extends the core combination with model_partial_name and seed."""

    model_full: str


@dataclass
class NoiseAnalysisModes(BaseAnalysisModes[NoiseAnalysisCombination]):
    """Extends the core modes with model_partial_name and seed lists.

    Subclass the generic BaseAnalysisModes with C=CheckpointAnalysisCombination.
    """

    model_full_list: list[str] = field(
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
        self.model_full_list = concatenated_df["model_full"].unique().tolist()

    def all_combinations(
        self,
    ) -> list[NoiseAnalysisCombination]:
        """Return all possible combinations."""
        # First get the base lists from BaseAnalysisModes
        data_full_list: list[str] = self.data_full_list
        data_subsampling_split_list: list[str] = self.data_subsampling_split_list
        data_subsampling_sampling_mode_list: list[str] = self.data_subsampling_sampling_mode_list
        embedding_data_handler_mode_list: list[str] = self.embedding_data_handler_mode_list

        # Then add the two new lists
        model_full_list: list[str] = self.model_full_list

        return [
            NoiseAnalysisCombination(*combo)
            for combo in product(
                data_full_list,
                data_subsampling_split_list,
                data_subsampling_sampling_mode_list,
                embedding_data_handler_mode_list,
                model_full_list,
            )
        ]

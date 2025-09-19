from dataclasses import dataclass, field
from typing import Generic, TypeVar

import pandas as pd

from topollm.config_classes.constants import NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS


@dataclass
class BaseAnalysisCombination:
    """Configuration for a single combination focusing on data & embedding modes only."""

    data_full: str
    data_subsampling_split: str
    data_subsampling_sampling_mode: str
    embedding_data_handler_mode: str


# Define a type variable C, bound to BaseAnalysisCombination
C = TypeVar(
    "C",
    bound=BaseAnalysisCombination,
)


@dataclass
class BaseAnalysisModes(Generic[C]):
    """Base configuration focusing on data & embedding modes."""

    data_full_list: list[str] = field(
        default_factory=list,
    )
    data_subsampling_split_list: list[str] = field(
        default_factory=lambda: [
            "train",
            "validation",
            "test",
        ],
    )
    data_subsampling_sampling_mode_list: list[str] = field(
        default_factory=list,
    )
    embedding_data_handler_mode_list: list[str] = field(
        default_factory=list,
    )

    common_filters_dict: dict = field(
        default_factory=lambda: {
            "data_subsampling_number_of_samples": 10_000,
            "data_prep_sampling_method": "random",
            "data_prep_sampling_samples": 150_000,
            NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS["local_estimates_dedup"]: "array_deduplicator",
            "local_estimates_samples": 60_000,
            "n_neighbors": 128,
        },
    )

    def from_concatenated_df(
        self,
        concatenated_df: pd.DataFrame,
    ) -> None:
        """Populate lists from columns in the provided DataFrame.

        Only the columns relevant to this 'base' iteration are used.
        """
        self.data_full_list = concatenated_df["data_full"].unique().tolist()
        self.data_subsampling_sampling_mode_list = concatenated_df["data_subsampling_sampling_mode"].unique().tolist()
        self.embedding_data_handler_mode_list = concatenated_df["embedding_data_handler_mode"].unique().tolist()

    def all_combinations(self) -> list[C]:
        """Return a list of combination objects of type C.

        The base version won't actually instantiate 'C'; it requires a subclass to specify how.
        """
        # This base method is optional (abstract method if needed).
        # If you do implement it, you need a way to construct 'C' objects, possibly via a factory method.
        # For now, we pretend this base method is just conceptual or abstract.
        msg = "Subclasses must implement all_combinations() to return List[C]"
        raise NotImplementedError(
            msg,
        )

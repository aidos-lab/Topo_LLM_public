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

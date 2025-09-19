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

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

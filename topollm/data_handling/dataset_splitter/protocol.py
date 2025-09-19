"""Protocol for splitting a dataset dict."""

from typing import Protocol

import datasets


class DatasetSplitter(Protocol):
    """Protocol for splitting a dataset."""

    def split_dataset(
        self,
        dataset_dict: datasets.DatasetDict,
    ) -> datasets.DatasetDict:
        """Create the split or rearrange the split."""
        ...  # pragma: no cover

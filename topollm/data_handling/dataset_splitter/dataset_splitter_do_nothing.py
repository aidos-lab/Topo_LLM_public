"""Do nothing with the dataset dict."""

import logging

import datasets

from topollm.typing.enums import Verbosity

logger: logging.Logger = logging.getLogger(
    name=__name__,
)


class DatasetSplitterDoNothing:
    """Do nothing with the dataset dict."""

    def __init__(
        self,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = logger,
    ) -> None:
        """Initialize the dataset splitter."""
        self.verbosity: Verbosity = verbosity
        self.logger: logging.Logger = logger

    def split_dataset(
        self,
        dataset_dict: datasets.DatasetDict,
    ) -> datasets.DatasetDict:
        """Return the dataset_dict unchanged."""
        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info("Returning unchanged dataset_dict.")
        return dataset_dict

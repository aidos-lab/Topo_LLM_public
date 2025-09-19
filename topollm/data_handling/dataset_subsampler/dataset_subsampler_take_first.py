"""Do nothing with the dataset dict."""

import logging

import datasets

from topollm.data_handling.dataset_subsampler.truncate_dataset import (
    truncate_dataset_with_maximum_the_actual_number_of_samples,
)
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


class DatasetSubsamplerTakeFirst:
    """Take the first entries from the dataset."""

    def __init__(
        self,
        number_of_samples: int,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Initialize the subsampler."""
        self.number_of_samples: int = number_of_samples

        self.verbosity: Verbosity = verbosity
        self.logger: logging.Logger = logger

    def subsample_dataset(
        self,
        dataset: datasets.Dataset,
    ) -> datasets.Dataset:
        """Take first sequences from the dataset."""
        # Truncate the dataset to the specified number of samples
        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg=f"Subsampling dataset to {self.number_of_samples = } samples (or how many are available) ...",  # noqa: G004 - low overhead
            )
        subsampled_dataset: datasets.Dataset = truncate_dataset_with_maximum_the_actual_number_of_samples(
            dataset=dataset,
            number_of_samples=self.number_of_samples,
            logger=self.logger,
        )
        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg=f"Subsampling dataset to {self.number_of_samples = } samples (or how many are available) DONE",  # noqa: G004 - low overhead
            )

        return subsampled_dataset

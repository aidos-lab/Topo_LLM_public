"""Helper function for truncating a dataset to a specified number of samples."""

import logging

import datasets

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def truncate_dataset_with_maximum_the_actual_number_of_samples(
    dataset: datasets.Dataset,
    number_of_samples: int,
    logger: logging.Logger = default_logger,
) -> datasets.Dataset:
    """Truncate the dataset to the specified number of samples.

    If `number_of_samples` is -1, all samples are used.
    If `number_of_samples` is greater than 0, the specified number of samples is used,
    unless the dataset has fewer samples, in which case all samples are used.
    """
    if number_of_samples == -1:
        # Use all samples
        subsampled_dataset: datasets.Dataset = dataset
    elif number_of_samples > 0:
        if number_of_samples > len(dataset):
            logger.warning(
                msg=f"Requested {number_of_samples = } samples, "  # noqa: G004 - low overhead
                f"but dataset only has {len(dataset) = } samples.",
            )

            actual_number_of_samples: int = len(dataset)

            logger.info(
                msg=f"Using {actual_number_of_samples = } samples instead.",  # noqa: G004 - low overhead
            )
        else:
            actual_number_of_samples: int = number_of_samples

        # Use only the specified number of samples
        subsampled_dataset = dataset.select(
            indices=range(actual_number_of_samples),
        )
    else:
        msg: str = f"Expected {number_of_samples = } to be -1 or a positive integer"
        raise ValueError(
            msg,
        )

    return subsampled_dataset

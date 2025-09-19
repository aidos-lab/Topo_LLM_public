"""Log information about a dataset."""

import logging
import pprint

import datasets
import torch.utils.data

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def log_huggingface_dataset_info(
    dataset: datasets.Dataset,
    dataset_name: str = "dataset",
    num_samples_to_log: int = 5,
    logger: logging.Logger = default_logger,
) -> None:
    """Log information about the dataset."""
    logger.info(
        f"{dataset_name}.info:\n%s",  # noqa: G004 - low overhead
        pprint.pformat(object=dataset.info),
    )
    logger.info(
        msg=f"{dataset_name}.column_names:\n{pprint.pformat(object=dataset.column_names)}",  # noqa: G004 - low overhead
    )
    logger.info(
        msg=f"{dataset_name}:\n{pprint.pformat(object=dataset)}",  # noqa: G004 - low overhead
    )

    # Log the first and last few samples of the dataset
    logger.info(
        f"{dataset_name}[:{num_samples_to_log}]:\n"  # noqa: G004 - low overhead
        f"{dataset[:num_samples_to_log]}",  # Do not use pprint here, as it will not be readable
    )
    logger.info(
        f"{dataset_name}[-{num_samples_to_log}:]:\n"  # noqa: G004 - low overhead
        f"{dataset[-num_samples_to_log:]}",  # Do not use pprint here, as it will not be readable
    )


def log_torch_dataset_info(
    dataset: torch.utils.data.Dataset,
    dataset_name: str = "dataset",
    num_samples_to_log: int = 5,
    logger: logging.Logger = default_logger,
) -> None:
    """Log information about the dataset."""
    logger.info(
        msg=f"{dataset_name = }",  # noqa: G004 - low overhead
    )
    logger.info(
        msg=f"{dataset_name}:\n{pprint.pformat(dataset)}",  # noqa: G004 - low overhead
    )

    # Log the first and last few samples of the dataset.
    # Note that torch datasets do not necessarily support slicing, so we cannot use dataset[:num_samples_to_log].
    # We implement it as a for loop instead.
    for idx in range(
        num_samples_to_log,
    ):
        # Do not use pprint for 'dataset[idx]', as it will not be readable
        logger.info(
            msg=f"{dataset_name}[{idx}]:\n{(dataset[idx])}",  # noqa: G004 - low overhead
        )

    for idx in range(
        -num_samples_to_log,
        0,
    ):
        # Do not use pprint for 'dataset[idx]', as it will not be readable
        logger.info(
            msg=f"{dataset_name}[{idx}]:\n{(dataset[idx])}",  # noqa: G004 - low overhead
        )

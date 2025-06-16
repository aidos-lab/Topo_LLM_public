# Copyright 2024-2025
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_1 (author1@example.com)
# AUTHOR_2 (author2@example.com)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#


"""Factory function to instantiate dataset preparers."""

import logging
from typing import TYPE_CHECKING

from topollm.config_classes.data.data_config import DataConfig
from topollm.data_handling.dataset_filtering.factory import get_dataset_filter
from topollm.data_handling.dataset_preparer import (
    dataset_preparer_huggingface,
    dataset_preparer_setsumbt_dataloaders_processed,
    dataset_preparer_trippy_dataloaders_processed,
)
from topollm.data_handling.dataset_preparer.protocol import DatasetPreparer
from topollm.data_handling.dataset_splitter.factory import get_dataset_splitter
from topollm.data_handling.dataset_subsampler.factory import get_dataset_subsampler
from topollm.typing.enums import DatasetType, Verbosity

if TYPE_CHECKING:
    from topollm.data_handling.dataset_filtering.protocol import DatasetFilter
    from topollm.data_handling.dataset_splitter.protocol import DatasetSplitter
    from topollm.data_handling.dataset_subsampler.protocol import DatasetSubsampler

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def get_dataset_preparer(
    data_config: DataConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> DatasetPreparer:
    """Return a dataset preparer for the given dataset type."""
    dataset_filter: DatasetFilter = get_dataset_filter(
        data_config=data_config,
        verbosity=verbosity,
        logger=logger,
    )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Using {dataset_filter.__class__.__name__ = } as dataset filter.",  # noqa: G004 - low overhead
        )

    dataset_splitter: DatasetSplitter = get_dataset_splitter(
        data_splitting_config=data_config.data_splitting,
        verbosity=verbosity,
        logger=logger,
    )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Using {dataset_splitter.__class__.__name__ = } as dataset splitter.",  # noqa: G004 - low overhead
        )

    dataset_subsampler: DatasetSubsampler = get_dataset_subsampler(
        data_subsampling_config=data_config.data_subsampling,
        verbosity=verbosity,
        logger=logger,
    )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Using {dataset_subsampler.__class__.__name__ = } as dataset subsampler.",  # noqa: G004 - low overhead
        )

    match data_config.dataset_type:
        case DatasetType.HUGGINGFACE_DATASET | DatasetType.HUGGINGFACE_DATASET_NAMED_ENTITY:
            result = dataset_preparer_huggingface.DatasetPreparerHuggingface(
                data_config=data_config,
                dataset_filter=dataset_filter,
                dataset_splitter=dataset_splitter,
                dataset_subsampler=dataset_subsampler,
                verbosity=verbosity,
                logger=logger,
            )
        case DatasetType.SETSUMBT_DATALOADERS_PROCESSED:
            result = dataset_preparer_setsumbt_dataloaders_processed.DatasetPreparerSetSUMBTDataloadersProcessed(
                data_config=data_config,
                dataset_filter=dataset_filter,
                dataset_splitter=dataset_splitter,
                dataset_subsampler=dataset_subsampler,
                verbosity=verbosity,
                logger=logger,
            )
        case DatasetType.TRIPPY_DATALOADERS_PROCESSED | DatasetType.TRIPPY_R_DATALOADERS_PROCESSED:
            result = dataset_preparer_trippy_dataloaders_processed.DatasetPreparerTrippyDataloadersProcessed(
                data_config=data_config,
                dataset_filter=dataset_filter,
                dataset_splitter=dataset_splitter,
                dataset_subsampler=dataset_subsampler,
                verbosity=verbosity,
                logger=logger,
            )
        case _:
            msg: str = f"Unsupported {data_config.dataset_type = }"
            raise ValueError(
                msg,
            )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Using {result.__class__.__name__ = } as dataset preparer.",  # noqa: G004 - low overhead
        )

    return result

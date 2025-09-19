"""Factory function to instantiate dataset splitters."""

import logging

from topollm.config_classes.data.data_splitting_config import DataSplittingConfig
from topollm.data_handling.dataset_splitter import dataset_splitter_do_nothing, dataset_splitter_proportions
from topollm.data_handling.dataset_splitter.protocol import DatasetSplitter
from topollm.typing.enums import DataSplitMode, Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def get_dataset_splitter(
    data_splitting_config: DataSplittingConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> DatasetSplitter:
    """Return a dataset splitter."""
    if data_splitting_config.data_splitting_mode == DataSplitMode.DO_NOTHING:
        result = dataset_splitter_do_nothing.DatasetSplitterDoNothing(
            verbosity=verbosity,
            logger=logger,
        )
    elif data_splitting_config.data_splitting_mode == DataSplitMode.PROPORTIONS:
        result = dataset_splitter_proportions.DatasetSplitterProportions(
            proportions=data_splitting_config.proportions,
            split_shuffle=data_splitting_config.split_shuffle,
            split_seed=data_splitting_config.split_seed,
            verbosity=verbosity,
            logger=logger,
        )
    else:
        msg: str = f"Unsupported {data_splitting_config.data_splitting_mode = }"
        raise ValueError(msg)

    return result

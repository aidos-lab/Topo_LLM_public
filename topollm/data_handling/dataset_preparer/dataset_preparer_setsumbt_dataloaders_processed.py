"""Prepare a dataset from huggingface datasets."""

import logging
import pathlib

import datasets
import torch

from topollm.config_classes.data.data_config import DataConfig
from topollm.data_handling.dataset_filtering.protocol import DatasetFilter
from topollm.data_handling.dataset_splitter.protocol import DatasetSplitter
from topollm.data_handling.dataset_subsampler.protocol import DatasetSubsampler
from topollm.logging.log_dataset_info import log_huggingface_dataset_info
from topollm.task_performance_analysis.setsumbt.stack_tensors_from_dialogues_and_filter_fully_padded_utterances import (
    stack_tensors_from_dialogues_and_filter_fully_padded_utterances,
)
from topollm.typing.enums import DataSplitMode, Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


class DatasetPreparerSetSUMBTDataloadersProcessed:
    """Prepares a dataset from the saved SetSUMBT dataloaders processed."""

    def __init__(
        self,
        data_config: DataConfig,
        dataset_filter: DatasetFilter,
        dataset_splitter: DatasetSplitter,
        dataset_subsampler: DatasetSubsampler,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Initialize the dataset preparer."""
        self.data_config: DataConfig = data_config

        self.dataset_filter: DatasetFilter = dataset_filter
        self.dataset_splitter: DatasetSplitter = dataset_splitter
        self.dataset_subsampler: DatasetSubsampler = dataset_subsampler

        self.verbosity: Verbosity = verbosity
        self.logger: logging.Logger = logger

        self.dataset_length: int = -1

    def __len__(
        self,
    ) -> int:
        """Return the number of samples in the dataset."""
        if not hasattr(
            self,
            "dataset_length",
        ):
            msg = "The dataset length is not available."
            raise ValueError(
                msg,
            )

        if self.dataset_length < 0:
            msg = "The dataset length was not properly set."
            raise ValueError(
                msg,
            )

        return self.dataset_length

    def load_dataloaders_processed_from_disk(
        self,
    ) -> dict:
        """Load the dataloaders processed from disk."""
        dataloader_processed_path = pathlib.Path(
            self.data_config.dataset_path,
        )

        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg=f"Loading from {dataloader_processed_path = } ...",  # noqa: G004 - low overhead
            )

        dataloader_processed = torch.load(
            f=dataloader_processed_path,
        )

        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg=f"Loading from {dataloader_processed_path = } DONE",  # noqa: G004 - low overhead
            )

        return dataloader_processed

    def convert_dataloader_processed_to_dataset(
        self,
        dataloader_processed: dict,
    ) -> datasets.Dataset:
        """Convert the dataloader processed to a dataset."""
        dataloader_stacked: dict = stack_tensors_from_dialogues_and_filter_fully_padded_utterances(
            dataloader_processed=dataloader_processed,
        )

        dataset: datasets.Dataset = datasets.Dataset.from_dict(
            mapping=dataloader_stacked,
        )

        return dataset

    def select_dataset(
        self,
        dataset: datasets.Dataset,
    ) -> datasets.Dataset:
        # Note: We do not apply a dataset splitter in this mode
        match self.data_config.data_splitting.data_splitting_mode:
            case DataSplitMode.DO_NOTHING:
                self.logger.info(
                    msg="Data splitting mode is DO_NOTHING. Not splitting the dataset.",
                )
            case _:
                msg: str = (
                    f"Unsupported data splitting mode "
                    f"{self.data_config.data_splitting.data_splitting_mode = } for this dataset."
                )
                raise ValueError(
                    msg,
                )

        # # # #
        # Apply the dataset subsampler to the dataset
        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg="Applying dataset subsampler ...",
            )
        subsampled_dataset: datasets.Dataset = self.dataset_subsampler.subsample_dataset(
            dataset=dataset,
        )
        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg="Applying dataset subsampler DONE",
            )

        self.dataset_length = len(subsampled_dataset)

        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg=f"{self.dataset_length = }",  # noqa: G004 - no overhead
            )
            log_huggingface_dataset_info(
                dataset=subsampled_dataset,
                dataset_name="subsampled_dataset",
                logger=self.logger,
            )

        return subsampled_dataset

    def prepare_dataset(
        self,
    ) -> datasets.Dataset:
        """Load and prepare a dataset."""
        dataloader_processed: dict = self.load_dataloaders_processed_from_disk()

        dataset: datasets.Dataset = self.convert_dataloader_processed_to_dataset(
            dataloader_processed=dataloader_processed,
        )

        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg="Note: No filtering is implemented in this dataset preparer.",
            )

        dataset: datasets.Dataset = self.select_dataset(
            dataset=dataset,
        )

        return dataset

"""Prepare a dataset from huggingface datasets."""

import logging

import datasets

from topollm.config_classes.data.data_config import DataConfig
from topollm.data_handling.dataset_filtering.protocol import DatasetFilter
from topollm.data_handling.dataset_splitter.protocol import DatasetSplitter
from topollm.data_handling.dataset_subsampler.protocol import DatasetSubsampler
from topollm.logging.log_dataset_info import log_huggingface_dataset_info
from topollm.typing.enums import DatasetsLoadingFunction, Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


class DatasetPreparerHuggingface:
    """Prepares a dataset from huggingface datasets."""

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

    def load_dataset_dict(
        self,
    ) -> datasets.DatasetDict:
        """Load the dataset based from huggingface datasets based on configuration."""
        # Note:
        # If you want to load datasets which were saved via `.save_to_disk()`,
        # you should use the `.load_from_disk()` method instead of the `.load_dataset()` method,
        # because that will directly use the arrow files and the given directory as cache.
        # See also:
        # https://github.com/huggingface/datasets/issues/6703
        match self.data_config.datasets_loading_function:
            case DatasetsLoadingFunction.LOAD_DATASET:
                dataset_dict = datasets.load_dataset(
                    path=self.data_config.dataset_path,
                    name=self.data_config.dataset_name,
                    data_dir=self.data_config.data_dir,  # type: ignore - type annotation for path
                    trust_remote_code=True,
                )
            case DatasetsLoadingFunction.LOAD_FROM_DISK:
                dataset_dict: datasets.Dataset | datasets.DatasetDict = datasets.load_from_disk(
                    dataset_path=self.data_config.dataset_path,
                )
            case _:
                msg: str = f"Unsupported datasets loading function: {self.data_config.datasets_loading_function = }."
                raise ValueError(
                    msg,
                )

        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg=f"dataset_dict:\n{dataset_dict}",  # noqa: G004 - low overhead
            )

        if not isinstance(
            dataset_dict,
            datasets.DatasetDict,
        ):
            msg: str = f"Expected {dataset_dict = } to be a datasets.DatasetDict, but got {type(dataset_dict) = }."
            raise TypeError(
                msg,
            )

        return dataset_dict

    def select_dataset(
        self,
        dataset_dict: datasets.DatasetDict,
    ) -> datasets.Dataset:
        if self.verbosity >= Verbosity.NORMAL:
            default_logger.info(
                "dataset_dict:\n%s",
                dataset_dict,
            )
            default_logger.info(
                msg="Applying dataset splitter ...",
            )

        # Apply the dataset splitter to the dataset
        new_dataset_dict: datasets.DatasetDict = self.dataset_splitter.split_dataset(
            dataset_dict=dataset_dict,
        )
        if self.verbosity >= Verbosity.NORMAL:
            default_logger.info(
                msg="Applying dataset splitter DONE.",
            )
            default_logger.info(
                "new_dataset_dict:\n%s",
                new_dataset_dict,
            )

        # # # #
        # Select the dataset split to use
        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg=f"Selecting dataset split {self.data_config.data_subsampling.split.value = } ...",  # noqa: G004 - low overhead
            )
        dataset: datasets.Dataset = new_dataset_dict[self.data_config.data_subsampling.split.value]
        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg=f"Selecting dataset split {self.data_config.data_subsampling.split.value = } DONE",  # noqa: G004 - low overhead
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
        dataset_dict: datasets.DatasetDict = self.load_dataset_dict()

        dataset_dict_filtered: datasets.DatasetDict = self.dataset_filter.filter_dataset_dict(
            dataset_dict=dataset_dict,
        )

        dataset: datasets.Dataset = self.select_dataset(
            dataset_dict=dataset_dict_filtered,
        )

        # Apply the dataset normalizer
        if self.data_config.normalization.apply_string_strip:
            dataset_normalized: datasets.Dataset = dataset.map(
                function=lambda x: {self.data_config.column_name: x[self.data_config.column_name].strip()},
                batched=False,
            )
        else:
            dataset_normalized: datasets.Dataset = dataset

        if self.verbosity >= Verbosity.NORMAL:
            log_huggingface_dataset_info(
                dataset=dataset_normalized,
                dataset_name="dataset_normalized",
                logger=self.logger,
            )

        return dataset_normalized

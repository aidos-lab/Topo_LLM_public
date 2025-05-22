# Copyright 2024-2025
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Matthias Ruppik (mail@ruppik.net)
# Julius von Rohrscheidt (julius.rohrscheidt@helmholtz-muenchen.de)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare a dataset from huggingface datasets."""

import logging

import datasets

from topollm.config_classes.data.data_config import DataConfig
from topollm.data_handling.dataset_filtering.protocol import DatasetFilter
from topollm.data_handling.dataset_splitter.protocol import DatasetSplitter
from topollm.data_handling.dataset_subsampler.protocol import DatasetSubsampler
from topollm.logging.log_dataset_info import log_huggingface_dataset_info
from topollm.typing.enums import Verbosity

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
        dataset_dict = datasets.load_dataset(
            path=self.data_config.dataset_path,
            name=self.data_config.dataset_name,
            data_dir=self.data_config.data_dir,  # type: ignore - type annotation for path
            trust_remote_code=True,
        )

        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                "dataset_dict:\n%s",
                dataset_dict,
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

# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (ruppik@hhu.de)
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
from topollm.data_handling.dataset_splitter.protocol import DatasetSplitter
from topollm.logging.log_dataset_info import log_huggingface_dataset_info

logger = logging.getLogger(__name__)


class DatasetPreparerHuggingface:
    """Prepares a dataset from huggingface datasets."""

    def __init__(
        self,
        data_config: DataConfig,
        dataset_splitter: DatasetSplitter,
        verbosity: int = 1,
        logger: logging.Logger = logger,
    ) -> None:
        """Initialize the dataset preparer."""
        self.data_config = data_config

        self.dataset_splitter = dataset_splitter

        self.verbosity = verbosity
        self.logger = logger

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
            raise ValueError(msg)

        if self.dataset_length < 0:
            msg = "The dataset length was not properly set."
            raise ValueError(msg)

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

        if self.verbosity >= 1:
            self.logger.info(
                "dataset_dict:\n%s",
                dataset_dict,
            )

        if not isinstance(
            dataset_dict,
            datasets.DatasetDict,
        ):
            msg = f"Expected {dataset_dict = } to be a datasets.DatasetDict"
            raise TypeError(msg)

        return dataset_dict

    def select_dataset(
        self,
        dataset_dict: datasets.DatasetDict,
    ) -> datasets.Dataset:
        if self.verbosity >= 1:
            logger.info(
                "dataset_dict:\n%s",
                dataset_dict,
            )
            logger.info("Applying dataset splitter ...")

        # Apply the dataset splitter to the dataset
        new_dataset_dict = self.dataset_splitter.split_dataset(
            dataset_dict=dataset_dict,
        )
        if self.verbosity >= 1:
            logger.info(
                "new_dataset_dict:\n%s",
                new_dataset_dict,
            )

        # Select the dataset split to use
        dataset: datasets.Dataset = new_dataset_dict[self.data_config.split]

        # Truncate the dataset to the specified number of samples
        if self.data_config.number_of_samples == -1:
            # Use all samples
            pass
        elif self.data_config.number_of_samples > 0:
            # Use only the specified number of samples
            dataset = dataset.select(
                indices=range(self.data_config.number_of_samples),
            )
        else:
            msg = f"Expected {self.data_config.number_of_samples = } to be -1 or a positive integer"
            raise ValueError(msg)

        self.dataset_length = len(dataset)

        if self.verbosity >= 1:
            self.logger.info(
                f"{self.dataset_length = }",  # noqa: G004 - no overhead
            )
            log_huggingface_dataset_info(
                dataset=dataset,
                dataset_name="dataset",
                logger=self.logger,
            )

        return dataset

    def prepare_dataset(
        self,
    ) -> datasets.Dataset:
        """Load and prepare a dataset."""
        dataset_dict = self.load_dataset_dict()

        dataset = self.select_dataset(
            dataset_dict=dataset_dict,
        )

        return dataset

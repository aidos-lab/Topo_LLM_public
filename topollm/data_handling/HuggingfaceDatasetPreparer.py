# coding=utf-8
#
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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Imports

# Standard library imports
import logging

# Third party imports
import datasets

# Local imports
from topollm.config_classes.Configs import DataConfig
from topollm.logging.log_dataset_info import log_huggingface_dataset_info

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #



class HuggingfaceDatasetPreparer:
    """Prepares a dataset from huggingface datasets."""


    def __init__(
        self,
        data_config: DataConfig,
        verbosity: int = 1,
        logger: logging.Logger = logging.getLogger(__name__),
    ) -> None:
        self.data_config = data_config

        self.verbosity = verbosity
        self.logger = logger

        self.dataset_length: int = -1

        return None

    def load_dataset_dict(
        self,
    ) -> datasets.DatasetDict:
        """Loads the dataset based from huggingface datasets based on configuration."""
        dataset_dict = datasets.load_dataset(
            path=self.data_config.dataset_path,
            name=self.data_config.dataset_name,
            data_dir=self.data_config.data_dir, # type: ignore
            trust_remote_code=True,
        )

        if self.verbosity >= 1:
            self.logger.info(
                f"{dataset_dict = }",
            )

        if not isinstance(
            dataset_dict,
            datasets.DatasetDict,
        ):
            raise ValueError(
                f"Expected {dataset_dict = } " f"to be a {datasets.DatasetDict = }"
            )

        return dataset_dict

    def select_dataset(
        self,
        dataset_dict: datasets.DatasetDict,
    ) -> datasets.Dataset:
        # Select the dataset split to use
        dataset: datasets.Dataset = dataset_dict[
            self.data_config.split
        ]

        # Truncate the dataset to the specified number of samples
        dataset = dataset.select(
            indices=range(self.data_config.number_of_samples),
        )

        self.dataset_length = len(dataset)

        if self.verbosity >= 1:
            self.logger.info(
                f"{self.dataset_length = }",
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
        """
        Loads and prepares a dataset.
        """

        dataset_dict = self.load_dataset_dict()

        dataset = self.select_dataset(
            dataset_dict=dataset_dict,
        )

        return dataset
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

"""Split a dataset into training, validation, and test set with given proportions."""

import logging

import datasets

logger = logging.getLogger(__name__)


class DatasetSplitterProportions:
    """Splits a dataset into training, validation, and test set with given proportions."""

    def __init__(
        self,
        train_proportion: float,
        validation_proportion: float,
        test_proportion: float,
        verbosity: int = 1,
        logger: logging.Logger = logger,
    ) -> None:
        """Initialize the dataset splitter."""
        self.train_proportion = train_proportion
        self.validation_proportion = validation_proportion
        self.test_proportion = test_proportion

        self.verbosity = verbosity
        self.logger = logger

    def split_dataset(
        self,
        dataset_dict: datasets.DatasetDict,
    ) -> datasets.DatasetDict:
        """Split the dataset into training, validation, and test set with given proportions."""
        dataset = dataset_dict["train"]

        # Split the dataset into train and the rest
        train_and_rest = dataset.train_test_split(
            test_size=self.validation_proportion + self.test_proportion,
            shuffle=False,
        )
        # Split the rest into validation and test
        normalized_test_proportion = self.test_proportion / (self.validation_proportion + self.test_proportion)
        validation_and_test = train_and_rest["test"].train_test_split(
            test_size=normalized_test_proportion,
            shuffle=False,
        )

        # Gather everything into a single DatasetDict
        train_validation_test_dataset = datasets.DatasetDict(
            {
                "train": train_and_rest["train"],
                "validation": validation_and_test["train"],
                "test": validation_and_test["test"],
            },
        )

        return train_validation_test_dataset

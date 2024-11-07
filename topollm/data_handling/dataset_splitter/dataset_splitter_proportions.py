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

from topollm.config_classes.data.data_splitting_config import Proportions
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


class DatasetSplitterProportions:
    """Splits a dataset into training, validation, and test set with given proportions."""

    def __init__(
        self,
        proportions: Proportions,
        *,
        split_shuffle: bool = False,
        split_seed: int | None = None,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Initialize the dataset splitter."""
        self.proportions: Proportions = proportions
        self.split_shuffle: bool = split_shuffle
        self.split_seed: int | None = split_seed

        self.verbosity: Verbosity = verbosity
        self.logger: logging.Logger = logger

    def split_dataset(
        self,
        dataset_dict: datasets.DatasetDict,
    ) -> datasets.DatasetDict:
        """Split the dataset into training, validation, and test set with given proportions."""
        dataset: datasets.Dataset = dataset_dict["train"]
        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg=f"Length of train for the original dict:\n{len(dataset) = }",  # noqa: G004 - low overhead
            )

        # Split the dataset into train and the remainder/leftover
        train_and_leftover: datasets.DatasetDict = dataset.train_test_split(
            test_size=self.proportions.validation + self.proportions.test,
            shuffle=self.split_shuffle,
            seed=self.split_seed,
        )
        # Split the leftover into validation and test
        normalized_test_proportion = self.proportions.test / (self.proportions.validation + self.proportions.test)
        validation_and_test: datasets.DatasetDict = train_and_leftover["test"].train_test_split(
            test_size=normalized_test_proportion,
            shuffle=self.split_shuffle,
            seed=self.split_seed,
        )

        # Gather everything into a single DatasetDict
        train_validation_test_dataset = datasets.DatasetDict(
            {
                "train": train_and_leftover["train"],
                "validation": validation_and_test["train"],
                "test": validation_and_test["test"],
            },
        )

        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg=f"Length of new dict entry:\n{len(train_validation_test_dataset['train']) = }",  # noqa: G004 - low overhead
            )
            self.logger.info(
                msg=f"Length of new dict entry:\n{len(train_validation_test_dataset['validation']) = }",  # noqa: G004 - low overhead
            )
            self.logger.info(
                msg=f"Length of new dict entry:\n{len(train_validation_test_dataset['test']) = }",  # noqa: G004 - low overhead
            )

        return train_validation_test_dataset

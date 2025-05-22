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

from topollm.data_handling.dataset_subsampler.truncate_dataset import (
    truncate_dataset_with_maximum_the_actual_number_of_samples,
)
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


class DatasetSubsamplerRandom:
    """Sample random sequences from the dataset."""

    def __init__(
        self,
        number_of_samples: int,
        sampling_seed: int,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Initialize the dataset splitter."""
        self.number_of_samples: int = number_of_samples
        self.sampling_seed: int = sampling_seed

        self.verbosity: Verbosity = verbosity
        self.logger: logging.Logger = logger

    def subsample_dataset(
        self,
        dataset: datasets.Dataset,
    ) -> datasets.Dataset:
        """Take random sequences from the dataset."""
        # Tutorial for taking a random dataset subsample:
        # https://huggingface.co/learn/nlp-course/en/chapter5/3

        # Shuffle the dataset first
        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg=f"Shuffling dataset with {self.sampling_seed = } ...",  # noqa: G004 - low overhead
            )
        dataset_shuffled: datasets.Dataset = dataset.shuffle(
            seed=self.sampling_seed,
        )
        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg=f"Shuffling dataset with {self.sampling_seed = } DONE",  # noqa: G004 - low overhead
            )

        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg=f"Subsampling dataset to {self.number_of_samples = } samples (or how many are available) ...",  # noqa: G004 - low overhead
            )
        dataset_shuffled_subsampled: datasets.Dataset = truncate_dataset_with_maximum_the_actual_number_of_samples(
            dataset=dataset_shuffled,
            number_of_samples=self.number_of_samples,
            logger=self.logger,
        )
        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg=f"Subsampling dataset to {self.number_of_samples = } samples (or how many are available) DONE",  # noqa: G004 - low overhead
            )

        return dataset_shuffled_subsampled

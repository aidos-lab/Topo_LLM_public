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
        dataset_shuffled: datasets.Dataset = dataset.shuffle(
            seed=self.sampling_seed,
        )

        if self.number_of_samples == -1:
            # Use all samples
            dataset_shuffled_subsampled: datasets.Dataset = dataset_shuffled
        elif self.number_of_samples > 0:
            dataset_shuffled_subsampled = dataset_shuffled.select(
                indices=range(self.number_of_samples),
            )
        else:
            msg: str = f"Expected {self.number_of_samples = } to be -1 or a positive integer"
            raise ValueError(
                msg,
            )

        return dataset_shuffled_subsampled

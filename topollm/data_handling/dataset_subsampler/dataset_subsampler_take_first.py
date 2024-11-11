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

"""Do nothing with the dataset dict."""

import logging

import datasets

from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


class DatasetSubsamplerTakeFirst:
    """Take the first entries from the dataset."""

    def __init__(
        self,
        number_of_samples: int,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Initialize the subsampler."""
        self.number_of_samples: int = number_of_samples

        self.verbosity: Verbosity = verbosity
        self.logger: logging.Logger = logger

    def subsample_dataset(
        self,
        dataset: datasets.Dataset,
    ) -> datasets.Dataset:
        """Take first sequences from the dataset."""
        # Truncate the dataset to the specified number of samples
        if self.number_of_samples == -1:
            # Use all samples
            pass
        elif self.number_of_samples > 0:
            # Use only the specified number of samples
            dataset = dataset.select(
                indices=range(self.number_of_samples),
            )
        else:
            msg: str = f"Expected {self.number_of_samples = } to be -1 or a positive integer"
            raise ValueError(msg)

        return dataset

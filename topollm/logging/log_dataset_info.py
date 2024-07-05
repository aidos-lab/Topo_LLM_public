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

import logging
import pprint

import datasets
import torch.utils.data

default_logger = logging.getLogger(__name__)


def log_huggingface_dataset_info(
    dataset: datasets.Dataset,
    dataset_name: str = "dataset",
    num_samples_to_log: int = 5,
    logger: logging.Logger = default_logger,
) -> None:
    """Log information about the dataset."""
    logger.info(
        f"{dataset_name}.info:\n%s",  # noqa: G004 - low overhead
        pprint.pformat(dataset.info),
    )
    logger.info(
        f"{dataset_name}.column_names:\n" f"{pprint.pformat(dataset.column_names)}",
    )
    logger.info(
        f"{dataset_name}:\n" f"{pprint.pformat(dataset)}",
    )

    # Log the first and last few samples of the dataset
    logger.info(
        f"{dataset_name}[:{num_samples_to_log}]:\n"
        f"{dataset[:num_samples_to_log]}",  # Do not use pprint here, as it will not be readable
    )
    logger.info(
        f"{dataset_name}[-{num_samples_to_log}:]:\n"
        f"{dataset[-num_samples_to_log:]}",  # Do not use pprint here, as it will not be readable
    )


def log_torch_dataset_info(
    dataset: torch.utils.data.Dataset,
    dataset_name: str = "dataset",
    num_samples_to_log: int = 5,
    logger: logging.Logger = default_logger,
) -> None:
    """Log information about the dataset."""
    logger.info(
        f"{dataset_name = }",
    )
    logger.info(
        f"{dataset_name}:\n" f"{pprint.pformat(dataset)}",
    )

    # Log the first and last few samples of the dataset.
    # Note that torch datasets do not necessarily support slicing, so we cannot use dataset[:num_samples_to_log].
    # We implement it as a for loop instead.
    for idx in range(
        num_samples_to_log,
    ):
        logger.info(
            f"{dataset_name}[{idx}]:\n" f"{(dataset[idx])}",  # Do not use pprint here, as it will not be readable
        )

    for idx in range(
        -num_samples_to_log,
        0,
    ):
        logger.info(
            f"{dataset_name}[{idx}]:\n" f"{(dataset[idx])}",  # Do not use pprint here, as it will not be readable
        )

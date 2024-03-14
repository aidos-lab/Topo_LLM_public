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
import pprint

# Third party imports
import datasets

# Local imports

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def log_dataset_info(
    dataset: datasets.Dataset,
    dataset_name: str = "dataset",
    num_samples_to_log: int = 5,
    logger: logging.Logger = logging.getLogger(__name__),
) -> None:
    """Logs information about the dataset."""
    logger.info(
        f"{dataset_name}.info:\n" f"{pprint.pformat(dataset.info)}",
    )
    logger.info(
        f"{dataset_name}.column_names:\n" f"{pprint.pformat(dataset.column_names)}",
    )
    logger.info(
        f"{dataset_name}:\n" f"{pprint.pformat(dataset)}",
    )

    # Log the first and last few samples of the dataset
    logger.info(
        f"{dataset_name[:num_samples_to_log]}:\n"
        f"{dataset[:num_samples_to_log]}",  # Do not use pprint here, as it will not be readable
    )
    logger.info(
        f"{dataset[-num_samples_to_log:]}:\n",
        f"{dataset_name[-num_samples_to_log:]}",  # Do not use pprint here, as it will not be readable
    )

    return None

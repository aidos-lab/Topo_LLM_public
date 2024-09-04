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
import random

import datasets
import pandas as pd
from datasets import ClassLabel, Sequence

from topollm.logging.log_dataframe_info import log_dataframe_info

default_logger = logging.getLogger(__name__)


def select_random_elements_and_create_dataframe_with_classlabel(
    dataset: datasets.Dataset,
    num_examples: int = 10,
) -> pd.DataFrame:
    """Select random elements from a dataset and create a pandas DataFrame from them."""
    if num_examples > len(dataset):
        msg = "Can't pick more elements than there are in the dataset."
        raise ValueError(msg)

    picks: list[int] = []
    for _ in range(num_examples):
        pick = random.randint(  # noqa: S311 - not used for security/cryptographic purposes
            0,
            len(dataset) - 1,
        )
        while pick in picks:
            pick = random.randint(  # noqa: S311 - not used for security/cryptographic purposes
                0,
                len(dataset) - 1,
            )
        picks.append(pick)

    selected_elements = [dataset[i] for i in picks]
    selected_elements_df = pd.DataFrame(
        selected_elements,
    )

    for column_name, feature_type in dataset.features.items():
        feature_type: datasets.features.features.Value

        if isinstance(
            feature_type,
            ClassLabel,
        ):
            new_column_name = f"{column_name}_transformed"
            selected_elements_df[new_column_name] = selected_elements_df[column_name].transform(
                lambda i, feature_type=feature_type: feature_type.names[i],
            )
        elif isinstance(
            feature_type,
            Sequence,
        ) and isinstance(
            feature_type.feature,
            ClassLabel,
        ):
            new_column_name = f"{column_name}_transformed"
            selected_elements_df[new_column_name] = selected_elements_df[column_name].transform(
                lambda x, feature_type=feature_type: [feature_type.feature.names[i] for i in x],
            )

    return selected_elements_df


def log_selected_dataset_elements_info(
    dataset: datasets.Dataset,
    dataset_name: str = "selected_dataset_elements",
    num_examples: int = 20,
    logger: logging.Logger = default_logger,
) -> None:
    """Select random elements from a dataset and log information about them."""
    try:
        selected_elements_df = select_random_elements_and_create_dataframe_with_classlabel(
            dataset=dataset,
            num_examples=num_examples,
        )
        log_dataframe_info(
            df=selected_elements_df,
            df_name=dataset_name,
            logger=logger,
        )
    except ValueError as e:
        logger.warning(e)

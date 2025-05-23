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

"""Remove padding and extra tokens from the data."""

import logging

import numpy as np
import pandas as pd

from topollm.config_classes.data_processing_column_names.data_processing_column_names import DataProcessingColumnNames
from topollm.config_classes.embeddings_data_prep.filter_tokens_config import FilterTokensConfig
from topollm.embeddings_data_prep.get_token_ids_from_filter_tokens_config import get_token_ids_from_filter_tokens_config
from topollm.typing.enums import Verbosity
from topollm.typing.types import TransformersTokenizer

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)
default_data_processing_column_names = DataProcessingColumnNames()


def remove_padding_and_extra_tokens(
    full_df: pd.DataFrame,
    tokenizer: TransformersTokenizer,
    filter_tokens_config: FilterTokensConfig,
    data_processing_column_names: DataProcessingColumnNames = default_data_processing_column_names,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> tuple[
    np.ndarray,
    pd.DataFrame,
]:
    """Remove padding and extra tokens from the data."""
    token_ids_to_filter: list[int] = get_token_ids_from_filter_tokens_config(
        tokenizer=tokenizer,
        filter_tokens_config=filter_tokens_config,
        verbosity=verbosity,
        logger=logger,
    )

    # Remove the specified tokens from the data
    filtered_df = full_df[
        ~full_df[data_processing_column_names.token_id].isin(
            token_ids_to_filter,
        )
    ]

    # Construct the array from the filtered data
    #
    # arr_no_pad.shape:
    # (number of non-padding tokens in subsample, embedding dimension)
    filtered_array = np.array(
        list(filtered_df[data_processing_column_names.embedding_vectors]),
    )

    # Remove the array column from the DataFrame
    filtered_without_array_df = filtered_df.drop(
        columns=[data_processing_column_names.embedding_vectors],
    )

    return filtered_array, filtered_without_array_df

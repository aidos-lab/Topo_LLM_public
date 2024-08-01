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

"""Remove padding and extra tokens from the data."""

import logging

import numpy as np
import pandas as pd

from topollm.config_classes.embeddings_data_prep.filter_tokens_config import FilterTokensConfig
from topollm.embeddings_data_prep.get_token_ids_from_filter_tokens_config import get_token_ids_from_filter_tokens_config
from topollm.typing.enums import Verbosity
from topollm.typing.types import TransformersTokenizer

default_logger = logging.getLogger(__name__)


def remove_padding_and_extra_tokens(
    full_df: pd.DataFrame,
    tokenizer: TransformersTokenizer,
    filter_tokens_config: FilterTokensConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Remove padding and extra tokens from the data."""
    token_ids_to_filter: list[int] = get_token_ids_from_filter_tokens_config(
        tokenizer=tokenizer,
        filter_tokens_config=filter_tokens_config,
        verbosity=verbosity,
        logger=logger,
    )

    filtered_df = full_df[
        ~full_df["meta"].isin(
            token_ids_to_filter,
        )
    ]

    # arr_no_pad.shape:
    # (number of non-padding tokens in subsample, embedding dimension)
    arr_no_pad = np.array(
        list(filtered_df.arr),
    )

    # meta_no_pad.shape:
    # (number of non-padding tokens in subsample,)
    meta_no_pad = np.array(
        list(filtered_df.meta),
    )

    # sentence_idx_no_pad.shape:
    # (number of non-padding tokens in subsample,)
    sentence_idx_no_pad = np.array(
        list(filtered_df.sentence_idx),
    )

    return arr_no_pad, meta_no_pad, sentence_idx_no_pad

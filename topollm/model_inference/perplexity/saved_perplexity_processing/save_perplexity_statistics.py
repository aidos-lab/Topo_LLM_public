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
import pathlib

import pandas as pd

from topollm.config_classes.main_config import MainConfig
from topollm.embeddings_data_prep.get_token_ids_from_filter_tokens_config import get_token_ids_from_filter_tokens_config
from topollm.path_management.embeddings.protocol import EmbeddingsPathManager
from topollm.typing.enums import Verbosity
from topollm.typing.types import TransformersTokenizer

default_logger = logging.getLogger(__name__)


def save_perplexity_statistics(
    main_config: MainConfig,
    embeddings_path_manager: EmbeddingsPathManager,
    token_perplexities_df: pd.DataFrame,
    tokenizer: TransformersTokenizer,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> pd.DataFrame:
    """Save statistics about the perplexity dataframes into the perplexity directory.

    This function returns the token_perplexities_without_filtered_tokens_df.
    """
    token_ids_to_filter: list[int] = get_token_ids_from_filter_tokens_config(
        tokenizer=tokenizer,
        filter_tokens_config=main_config.embeddings_data_prep.filter_tokens,
        verbosity=verbosity,
        logger=logger,
    )

    token_perplexities_without_filtered_tokens_df: pd.DataFrame = token_perplexities_df[
        ~token_perplexities_df["token_id"].isin(
            token_ids_to_filter,
        )
    ]

    token_perplexities_without_special_tokens_df = token_perplexities_df[
        ~token_perplexities_df["token_id"].isin(
            tokenizer.all_special_ids,
        )
    ]

    # Save statistics about the perplexity dataframes into the perplexity directory
    perplexity_dir = embeddings_path_manager.perplexity_dir_absolute_path
    for current_df, current_df_description in [
        (
            token_perplexities_df,
            "token_perplexities_df",
        ),
        (
            token_perplexities_without_filtered_tokens_df,
            "token_perplexities_without_filtered_tokens_df",
        ),
        (
            token_perplexities_without_special_tokens_df,
            "token_perplexities_without_special_tokens_df",
        ),
    ]:
        current_df_statistics_save_path = pathlib.Path(
            perplexity_dir,
            f"{current_df_description}_statistics.csv",
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                f"{current_df_statistics_save_path = }",  # noqa: G004 - low overhead
            )
            logger.info(
                "Saving statistics to file ...",
            )

        current_df.describe().to_csv(
            path_or_buf=current_df_statistics_save_path,
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                "Saving statistics to file DONE",
            )

    return token_perplexities_without_filtered_tokens_df

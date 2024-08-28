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

"""Loading perplexity and local estimates."""

import logging

import huggingface_hub
import pandas as pd
import transformers
from huggingface_hub.errors import HFValidationError

from topollm.analysis.local_estimates.saving.local_estimates_containers import LocalEstimatesContainer
from topollm.analysis.local_estimates.saving.save_local_estimates import load_local_estimates
from topollm.config_classes.main_config import MainConfig
from topollm.logging.log_dataframe_info import log_dataframe_info
from topollm.model_handling.tokenizer.load_modified_tokenizer_from_main_config import (
    load_modified_tokenizer_from_main_config,
)
from topollm.model_inference.perplexity.saved_perplexity_processing.add_token_log_perplexity_column import (
    add_token_log_perplexity_column,
)
from topollm.model_inference.perplexity.saved_perplexity_processing.align_and_analyse.aligned_local_estimates_data_container import (
    AlignedLocalEstimatesDataContainer,
)
from topollm.model_inference.perplexity.saved_perplexity_processing.compare_columns import (
    compare_columns,
)
from topollm.model_inference.perplexity.saved_perplexity_processing.concatenate_results.convert_perplexity_results_list_to_dataframe import (
    convert_perplexity_results_list_to_dataframe,
)
from topollm.model_inference.perplexity.saved_perplexity_processing.load_perplexity_results import (
    load_perplexity_results,
)
from topollm.model_inference.perplexity.saved_perplexity_processing.save_perplexity_statistics import (
    save_perplexity_statistics,
)
from topollm.model_inference.perplexity.saving.save_concatenated_perplexity_results import (
    save_concatenated_perplexity_results,
)
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.typing.enums import Verbosity
from topollm.typing.types import PerplexityResultsList, TransformersTokenizer

default_logger = logging.getLogger(__name__)


def load_perplexity_and_local_estimates_and_align(
    main_config_for_perplexity: MainConfig,
    main_config_for_local_estimates: MainConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> AlignedLocalEstimatesDataContainer | None:
    """Load the perplexity results and the local estimates and align them.

    Returns
    -------
        aligned_df: The aligned dataframe containing the token perplexities and the local estimates.
        None: If the aligned dataframe could not be created.

    """
    # # # #
    # Get save paths
    perplexity_embeddings_path_manager = get_embeddings_path_manager(
        main_config=main_config_for_perplexity,
        logger=logger,
    )

    loaded_data: PerplexityResultsList = load_perplexity_results(
        embeddings_path_manager=perplexity_embeddings_path_manager,
        verbosity=verbosity,
        logger=logger,
    )

    # # # #
    # Convert the token perplexities to a pandas dataframe
    token_perplexities_df, token_perplexities_array = convert_perplexity_results_list_to_dataframe(
        loaded_data=loaded_data,
        verbosity=verbosity,
        logger=logger,
    )

    token_perplexities_df = add_token_log_perplexity_column(
        token_perplexities_df=token_perplexities_df,
    )

    # # # #
    # Save token perplexities as zarr array and pandas dataframe
    save_concatenated_perplexity_results(
        token_perplexities_df=token_perplexities_df,
        token_perplexities_array=token_perplexities_array,
        embeddings_path_manager=perplexity_embeddings_path_manager,
        verbosity=verbosity,
        logger=logger,
    )

    # # # # # # # # # # # # # # # # # # # #
    # Compute and save summary statistics

    tokenizer = load_tokenizer_with_fallback(
        main_config=main_config_for_perplexity,
        verbosity=verbosity,
        logger=logger,
    )

    token_perplexities_without_filtered_tokens_df = save_perplexity_statistics(
        main_config=main_config_for_perplexity,
        embeddings_path_manager=perplexity_embeddings_path_manager,
        token_perplexities_df=token_perplexities_df,
        tokenizer=tokenizer,
        verbosity=verbosity,
        logger=logger,
    )

    local_estimates_embeddings_path_manager = get_embeddings_path_manager(
        main_config=main_config_for_local_estimates,
        logger=logger,
    )

    local_estimates_container: LocalEstimatesContainer = load_local_estimates(
        embeddings_path_manager=local_estimates_embeddings_path_manager,
        verbosity=verbosity,
        logger=logger,
    )

    aligned_df: pd.DataFrame | None = create_aligned_df(
        local_estimates_container=local_estimates_container,
        token_perplexities_without_filtered_tokens_df=token_perplexities_without_filtered_tokens_df,
        verbosity=verbosity,
        logger=logger,
    )
    if aligned_df is None:
        logger.warning(
            "aligned_df is None. This function will return None.",
        )
        logger.warning(
            "Correlations between perplexities and local estimates cannot be computed.",
        )
        return None

    # Restrict to non-special tokens
    aligned_without_special_tokens_df: pd.DataFrame = aligned_df[
        ~aligned_df["token_id"].isin(
            tokenizer.all_special_ids,
        )
    ]

    data_container = AlignedLocalEstimatesDataContainer(
        main_config_for_perplexity=main_config_for_perplexity,
        main_config_for_local_estimates=main_config_for_local_estimates,
        aligned_df=aligned_df,
        aligned_without_special_tokens_df=aligned_without_special_tokens_df,
        verbosity=verbosity,
        logger=logger,
    )

    return data_container


def create_aligned_df(
    local_estimates_container: LocalEstimatesContainer,
    token_perplexities_without_filtered_tokens_df: pd.DataFrame,
    aligned_df_local_estimate_column_name: str = "local_estimate",
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> pd.DataFrame | None:
    """Create an aligned dataframe from the local estimates and the token perplexities."""
    local_estimates_meta_frame = local_estimates_container.results_meta_frame

    if local_estimates_meta_frame is None:
        logger.error(
            "local_estimates_meta_frame is None. The function will return None.",
        )
        return None

    if verbosity >= Verbosity.NORMAL:
        log_dataframe_info(
            df=local_estimates_meta_frame,
            df_name="local_estimates_meta_frame",
            logger=logger,
        )

    # Add the local estimates to the local_estimates_meta_frame
    local_estimates_meta_frame[aligned_df_local_estimate_column_name] = local_estimates_container.results_array_np

    corresponding_token_perplexities_df = token_perplexities_without_filtered_tokens_df.iloc[
        local_estimates_meta_frame["subsample_idx"]
    ]

    # Check that local_estimates_meta_frame["token_name"] and corresponding_token_perplexities_df["token_string"] agree
    discrepancies_token_string = compare_columns(
        df1=local_estimates_meta_frame,
        col1="token_name",
        df2=corresponding_token_perplexities_df,
        col2="token_string",
    )

    if not discrepancies_token_string.empty:
        logger.error(
            "local_estimates_meta_frame['token_name'] and "
            "corresponding_token_perplexities_df['token_string'] do not agree. "
            "The function will return None.",
        )
        return None

    # Check that local_estimates_meta_frame["token_id"] and corresponding_token_perplexities_df["token_id"] agree
    discrepancies_token_id = compare_columns(
        df1=local_estimates_meta_frame,
        col1="token_id",
        df2=corresponding_token_perplexities_df,
        col2="token_id",
    )

    if not discrepancies_token_id.empty:
        logger.error(
            "local_estimates_meta_frame['token_id'] and "
            "corresponding_token_perplexities_df['token_id'] do not agree. "
            "The function will return None.",
        )
        return None

    # Remove one instance of the 'token_id' column,
    # to avoid the `ValueError: cannot reindex on an axis with duplicate labels`
    corresponding_token_perplexities_df = corresponding_token_perplexities_df.drop(
        columns="token_id",
    )

    aligned_df = pd.concat(
        [
            corresponding_token_perplexities_df.reset_index(
                drop=True,
            ),
            local_estimates_meta_frame.reset_index(
                drop=True,
            ),
        ],
        axis=1,
    )

    return aligned_df


def load_tokenizer_with_fallback(
    main_config: MainConfig,
    fallback_pretrained_model_name_or_path: str = "roberta-base",
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> TransformersTokenizer:
    """Load the tokenizer with a fallback to `fallback_pretrained_model_name_or_path`."""
    try:
        tokenizer, _ = load_modified_tokenizer_from_main_config(
            main_config=main_config,
            verbosity=verbosity,
            logger=logger,
        )
    except (
        HFValidationError,
        FileNotFoundError,
        OSError,  # Add OSError to handle issues with paths or files
    ) as e:
        logger.exception(
            "Could not load the tokenizer from the provided configuration. Falling back to 'roberta-base'."
        )
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                fallback_pretrained_model_name_or_path,
            )
            logger.info(
                f"Successfully loaded fallback tokenizer {fallback_pretrained_model_name_or_path = }.",  # noqa: G004 - low overhead
            )
        except Exception as fallback_error:
            logger.exception(
                f"Fallback to {fallback_pretrained_model_name_or_path = } failed.",  # noqa: G004 - low overhead
            )
            raise fallback_error from e

    return tokenizer

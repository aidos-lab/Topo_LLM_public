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
import pathlib
from typing import TYPE_CHECKING

import huggingface_hub
import pandas as pd
import transformers

from topollm.analysis.local_estimates.saving.save_local_estimates import load_local_estimates
from topollm.config_classes.main_config import MainConfig
from topollm.embeddings_data_prep.get_token_ids_from_filter_tokens_config import get_token_ids_from_filter_tokens_config
from topollm.logging.log_dataframe_info import log_dataframe_info
from topollm.model_handling.tokenizer.load_modified_tokenizer_from_main_config import (
    load_modified_tokenizer_from_main_config,
)
from topollm.model_inference.perplexity.saved_perplexity_processing.add_token_log_perplexity_column import (
    add_token_log_perplexity_column,
)
from topollm.model_inference.perplexity.saved_perplexity_processing.compare_columns import (
    compare_columns,
)
from topollm.model_inference.perplexity.saved_perplexity_processing.concatenate_results.convert_perplexity_results_list_to_dataframe import (
    convert_perplexity_results_list_to_dataframe,
)
from topollm.model_inference.perplexity.saving.load_perplexity_containers_from_jsonl_files import (
    load_multiple_perplexity_containers_from_jsonl_files,
)
from topollm.model_inference.perplexity.saving.save_concatenated_perplexity_results import (
    save_concatenated_perplexity_results,
)
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.typing.enums import PerplexityContainerSaveFormat, Verbosity

if TYPE_CHECKING:
    from topollm.analysis.local_estimates.saving.local_estimates_containers import LocalEstimatesContainer
    from topollm.typing.types import PerplexityResultsList

default_logger = logging.getLogger(__name__)


def load_perplexity_and_local_estimates_and_align(
    main_config_for_perplexity: MainConfig,
    local_estimates_layer_indices: list[int] | None = None,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """

    # TODO: Document this function

    # TODO: Make this function return the loaded arrays so that we can use its output to compute the correlations between different setups
    """
    if local_estimates_layer_indices is None:
        local_estimates_layer_indices = [-1]

    # # # #
    # Get save paths
    perplexity_embeddings_path_manager = get_embeddings_path_manager(
        main_config=main_config_for_perplexity,
        logger=logger,
    )

    save_file_path_josnl = perplexity_embeddings_path_manager.get_perplexity_container_save_file_absolute_path(
        perplexity_container_save_format=PerplexityContainerSaveFormat.LIST_AS_JSONL,
    )

    loaded_data_list: list[PerplexityResultsList] = load_multiple_perplexity_containers_from_jsonl_files(
        path_list=[
            save_file_path_josnl,
        ],
        verbosity=verbosity,
        logger=logger,
    )

    # Since we are only loading one container, we can directly access the first element
    loaded_data: PerplexityResultsList = loaded_data_list[0]

    # # # #
    # Convert the token perplexities to a pandas dataframe
    token_perplexities_df, token_perplexities_array = convert_perplexity_results_list_to_dataframe(
        loaded_data=loaded_data,
        verbosity=verbosity,
        logger=logger,
    )

    add_token_log_perplexity_column(
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

    try:
        tokenizer, _ = load_modified_tokenizer_from_main_config(
            main_config=main_config_for_perplexity,
            verbosity=verbosity,
            logger=logger,
        )
    except (huggingface_hub.exceptions.ModelHubError, FileNotFoundError):
        logger.exception(
            "Could not load the tokenizer.",
        )
        # Use "roberta-base" as a fallback
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "roberta-base",
        )

    token_ids_to_filter: list[int] = get_token_ids_from_filter_tokens_config(
        tokenizer=tokenizer,
        filter_tokens_config=main_config_for_perplexity.embeddings_data_prep.filter_tokens,
        verbosity=verbosity,
        logger=logger,
    )

    token_perplexities_without_filtered_tokens_df: pd.DataFrame = token_perplexities_df[
        ~token_perplexities_df["token_id"].isin(token_ids_to_filter)
    ]

    token_perplexities_without_special_tokens_df = token_perplexities_df[
        ~token_perplexities_df["token_id"].isin(tokenizer.all_special_ids)
    ]

    # Save statistics about the perplexity dataframes into the perplexity directory
    perplexity_dir = perplexity_embeddings_path_manager.perplexity_dir_absolute_path
    for current_df, current_df_description in [
        (token_perplexities_df, "token_perplexities_df"),
        (token_perplexities_without_filtered_tokens_df, "token_perplexities_without_filtered_tokens_df"),
        (token_perplexities_without_special_tokens_df, "token_perplexities_without_special_tokens_df"),
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

    # # # # # # # # # # # # # # # # # # # #
    # Set the parameters so that the correct local estimates are loaded.
    # Note that we have to do this because the number of sequences for the perplexity computation
    # might be different from the number of sequences for the local estimates computation.

    # Make a configuration for the local estimates
    main_config_for_local_estimates = main_config_for_perplexity.model_copy(
        deep=True,
    )
    main_config_for_local_estimates.embeddings.embedding_extraction.layer_indices = local_estimates_layer_indices

    if main_config_for_local_estimates.data.dataset_description_string == "multiwoz21":
        main_config_for_local_estimates.data.number_of_samples = 3000
    else:
        main_config_for_local_estimates.data.number_of_samples = -1

    local_estimates_embeddings_path_manager = get_embeddings_path_manager(
        main_config=main_config_for_local_estimates,
        logger=logger,
    )

    local_estimates_container: LocalEstimatesContainer = load_local_estimates(
        embeddings_path_manager=local_estimates_embeddings_path_manager,
        verbosity=verbosity,
        logger=logger,
    )

    local_estimates_array_np = local_estimates_container.results_array_np

    # Create string with statistics
    local_estimates_statistics_string: str = (
        f"{local_estimates_array_np.shape = }\n"  # noqa: ISC003 - explicit string concatenation to avoid confusion
        + f"{local_estimates_array_np.mean() = }\n"
        + f"{local_estimates_array_np.std() = }\n"
        + f"{main_config_for_perplexity.data.number_of_samples = }\n"
        + f"{main_config_for_perplexity.embeddings.embedding_extraction.layer_indices = }\n"
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "local_estimates_statistics_string:\n%s",
            local_estimates_statistics_string,
        )

    # Save statistics to text file in the perplexity directory
    analyzed_data_save_directory: pathlib.Path = (
        perplexity_embeddings_path_manager.get_analyzed_data_dir_absolute_path()
    )
    analyzed_data_save_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    local_estimates_string_save_file_name: str = (
        "local_estimates_statistics"  # noqa: ISC003 - explicit string concatenation to avoid confusion
        + "_"
        + f"layer-{main_config_for_perplexity.embeddings.embedding_extraction.layer_indices}"
        + ".txt"
    )
    local_estimates_string_save_path = pathlib.Path(
        analyzed_data_save_directory,
        local_estimates_string_save_file_name,
    )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"{local_estimates_string_save_path = }",  # noqa: G004 - low overhead
        )
        logger.info(
            "Saving local_estimates_statistics_string to text file ...",
        )

    with local_estimates_string_save_path.open(
        mode="w",
    ) as f:
        f.write(
            local_estimates_statistics_string,
        )
        # Write the main_config to the file as well
        f.write(
            f"\n\nmain_config:\n{main_config_for_perplexity}\n",
        )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "Saving local_estimates_statistics_string to text file DONE",
        )

    local_estimates_meta_frame = local_estimates_container.results_meta_frame

    if local_estimates_meta_frame is None:
        logger.info("local_estimates_meta_frame is None.")
        logger.info("The function will return now without computing the correlations.")
        logger.warning("Correlations between perplexities and local estimates cannot be computed.")
        return

    if verbosity >= Verbosity.NORMAL:
        log_dataframe_info(
            df=local_estimates_meta_frame,
            df_name="local_estimates_meta_frame",
            logger=logger,
        )

    # Add the local estimates to the local_estimates_meta_frame
    local_estimates_meta_frame["local_estimate"] = local_estimates_array_np

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
            "corresponding_token_perplexities_df['token_string'] do not agree."
        )
        logger.error("The function will return now without computing the correlations.")
        logger.warning("Correlations between perplexities and local estimates cannot be computed.")
        return

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
            "corresponding_token_perplexities_df['token_id'] do not agree."
        )
        logger.error("The function will return now without computing the correlations.")
        logger.warning("Correlations between perplexities and local estimates cannot be computed.")
        return

    # Remove one instance of the 'token_id' column,
    # to avoid the `ValueError: cannot reindex on an axis with duplicate labels`
    corresponding_token_perplexities_df = corresponding_token_perplexities_df.drop(
        columns="token_id",
    )

    # Compute the correlation between the 'token_log_perplexity', and 'local_estimate'
    aligned_df = pd.concat(
        [
            corresponding_token_perplexities_df.reset_index(drop=True),
            local_estimates_meta_frame.reset_index(drop=True),
        ],
        axis=1,
    )

    # Restrict to non-special tokens
    aligned_without_special_tokens_df = aligned_df[
        ~aligned_df["token_id"].isin(
            tokenizer.all_special_ids,
        )
    ]

    # # # #
    # Saving aligned_df to csv file
    aligned_df_save_path = pathlib.Path(
        analyzed_data_save_directory,
        "aligned_df.csv",
    )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"{aligned_df_save_path = }",  # noqa: G004 - low overhead
        )
        logger.info(
            "Saving aligned_df to csv file ...",
        )
    aligned_df.to_csv(
        path_or_buf=aligned_df_save_path,
    )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "Saving aligned_df to csv file DONE",
        )

    aligned_without_special_tokens_df_save_path = pathlib.Path(
        analyzed_data_save_directory,
        "aligned_without_special_tokens_df.csv",
    )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"{aligned_without_special_tokens_df_save_path = }",  # noqa: G004 - low overhead
        )
        logger.info(
            "Saving aligned_without_special_tokens_df to csv file ...",
        )
    aligned_without_special_tokens_df.to_csv(
        path_or_buf=aligned_without_special_tokens_df_save_path,
    )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "Saving aligned_without_special_tokens_df to csv file DONE",
        )

    correlation_columns = [
        "token_perplexity",
        "token_log_perplexity",
        "local_estimate",
    ]
    only_correlation_columns_aligned_df = aligned_df[correlation_columns]

    if verbosity >= Verbosity.NORMAL:
        log_dataframe_info(
            df=aligned_df,
            df_name="aligned_df",
            logger=logger,
        )
        log_dataframe_info(
            df=only_correlation_columns_aligned_df,
            df_name="only_correlation_columns_aligned_df",
            logger=logger,
        )

    for method in [
        "pearson",
        "spearman",
        "kendall",
    ]:
        correlation_results_df = only_correlation_columns_aligned_df.corr(
            method=method,  # type: ignore - these methods are available
        )
        logger.info(
            f"Correlation using '{method = }':\n{correlation_results_df}",  # noqa: G004 - low overhead
        )
        logger.info(
            f"{correlation_results_df['local_estimate']['token_log_perplexity'] = }",  # noqa: G004 - low overhead
        )

        # Saving correlation_results_df to csv file
        correlation_results_df_save_path = pathlib.Path(
            analyzed_data_save_directory,
            f"correlation_results_{method}.csv",
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                f"{correlation_results_df_save_path = }",  # noqa: G004 - low overhead
            )
            logger.info(
                f"Saving correlation_results_df using '{method}' to csv file ...",  # noqa: G004 - low overhead
            )
        correlation_results_df.to_csv(
            path_or_buf=correlation_results_df_save_path,
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                f"Saving correlation_results_df using '{method}' to csv file DONE",  # noqa: G004 - low overhead
            )

    # TODO: Scatter plot of perplexity vs. local estimate

    # TODO: Plot histograms of perplexity and local estimate

    # # # # # # # # # # # # # # # # # # # #
    logger.info("Running script DONE")

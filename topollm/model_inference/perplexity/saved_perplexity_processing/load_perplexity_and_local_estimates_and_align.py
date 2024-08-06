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
from dataclasses import dataclass
from typing import TYPE_CHECKING

import huggingface_hub
import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd
import transformers

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
from topollm.model_inference.perplexity.saved_perplexity_processing.calculate_and_save_correlation_results import (
    calculate_and_save_correlation_results,
    extract_correlation_columns,
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
from topollm.model_inference.perplexity.saved_perplexity_processing.save_aligned_df_and_statistics import (
    save_aligned_df_and_statistics,
)
from topollm.model_inference.perplexity.saved_perplexity_processing.save_perplexity_statistics import (
    save_perplexity_statistics,
)
from topollm.model_inference.perplexity.saving.save_concatenated_perplexity_results import (
    save_concatenated_perplexity_results,
)
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.typing.enums import Verbosity
from topollm.typing.types import TransformersTokenizer

if TYPE_CHECKING:
    pass

default_logger = logging.getLogger(__name__)


def load_perplexity_and_local_estimates_and_align(
    main_config_for_perplexity: MainConfig,
    main_config_for_local_estimates: MainConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> pd.DataFrame | None:
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

    loaded_data = load_perplexity_results(
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

    tokenizer = load_tokenizer_with_roberta_fallback(
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

    aligned_df = create_aligned_df(
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

    # # # #
    # Saving aligned_df and statistics to csv files

    # Directory to save the analyzed data
    analyzed_data_save_directory: pathlib.Path = (
        local_estimates_embeddings_path_manager.get_analyzed_data_dir_absolute_path()
    )
    analyzed_data_save_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_aligned_df_and_statistics(
        aligned_df=aligned_df,
        aligned_without_special_tokens_df=aligned_without_special_tokens_df,
        analyzed_data_save_directory=analyzed_data_save_directory,
        verbosity=verbosity,
        logger=logger,
    )

    only_correlation_columns_aligned_df: pd.DataFrame = extract_correlation_columns(
        aligned_df=aligned_df,
        correlation_columns=None,
        verbosity=verbosity,
        logger=logger,
    )

    calculate_and_save_correlation_results(
        only_correlation_columns_aligned_df=only_correlation_columns_aligned_df,
        analyzed_data_save_directory=analyzed_data_save_directory,
        verbosity=verbosity,
        logger=logger,
    )

    # TODO(Ben): Implement saving of the histograms

    # # # #
    # Saving aligned_df and statistics to csv files

    # Manual settings for the columns
    manual_settings = {
        "token_perplexity": HistogramSettings(
            scale=(0, 10),
            bins=30,
        ),
        "token_log_perplexity": HistogramSettings(
            scale=(-10, 2),
            bins=50,
        ),
        "local_estimate": HistogramSettings(
            scale=(5, 15),
            bins=20,
        ),
    }

    # Automatic settings (select specific columns and use default bins)
    automatic_settings = {
        "token_perplexity": HistogramSettings(),
        "token_log_perplexity": HistogramSettings(),
        "local_estimate": HistogramSettings(),
    }

    # Plot histograms with automatic scaling (selected columns)
    figure = plot_histograms(
        df=aligned_df,
        settings=automatic_settings,
    )
    if figure is not None:
        plt.figure(figure)
        plt.show()

    # Plot histograms with manual scaling and configurable bins
    figure = plot_histograms(
        df=aligned_df,
        settings=manual_settings,
    )
    if figure is not None:
        plt.figure(figure)
        plt.show()

    # TODO(Ben): Scatter plot of perplexity vs. local estimate

    return aligned_df


@dataclass
class HistogramSettings:
    scale: tuple[float | None, float | None] | None = None
    bins: int | None = 30


def plot_histograms(
    df: pd.DataFrame,
    settings: dict[str, HistogramSettings] | None = None,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> matplotlib.figure.Figure | None:
    """Plot histograms for specified columns of a dataframe with optional manual scaling and configurable bins.

    Args:
    ----
        df: The dataframe containing the data.
        settings: Dictionary specifying the settings for each column.
            Each setting includes 'scale' (optional tuple of min and max for x-axis) and 'bins' (int for number of bins).
            If None, the histograms will be automatically scaled and use default bin count of 30.

    """
    columns_to_plot = df.select_dtypes(include="number").columns.tolist() if settings is None else list(settings.keys())

    num_columns = len(columns_to_plot)
    num_cols = 3  # Number of columns per row
    num_rows = (num_columns + num_cols - 1) // num_cols  # Calculate the number of rows needed

    # Debugging: Print the calculated values
    print(f"Number of columns to plot: {num_columns}")
    print(f"Number of rows: {num_rows}")
    print(f"Number of columns per row: {num_cols}")

    fig, axs = plt.subplots(
        num_rows,
        num_cols,
        figsize=(18, 6 * num_rows),
    )
    axs = axs.flatten()  # Flatten in case of multiple rows

    if columns_to_plot == []:
        logger.warning(
            "No columns to plot. The function will return None.",
        )
        return None

    i = 0  # Initialize index i to avoid error in case of empty columns_to_plot
    for i, column in enumerate(columns_to_plot):
        ax = axs[i]
        if settings and column in settings:
            scale = settings[column].scale
            bins = settings[column].bins
            if scale is not None:
                ax.hist(
                    df[column],
                    bins=bins,
                    alpha=0.7,
                    color="blue",
                    edgecolor="black",
                    range=scale,
                )
                ax.set_xlim(scale)  # Set the x-axis scale
            else:
                ax.hist(
                    df[column],
                    bins=bins,
                    alpha=0.7,
                    color="blue",
                    edgecolor="black",
                )
        else:
            ax.hist(
                df[column],
                bins=30,
                alpha=0.7,
                color="blue",
                edgecolor="black",
            )
        ax.set_title(f"Histogram of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")

    # Remove any unused subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    return fig


def save_plot(
    fig: matplotlib.figure.Figure,
    path: pathlib.Path,
) -> None:
    """Save the given plot to a specified path.

    Args:
    ----
        fig: The matplotlib figure object to save.
        path: The path where the plot should be saved.

    """
    fig.savefig(
        path,
    )


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


def load_tokenizer_with_roberta_fallback(
    main_config: MainConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> TransformersTokenizer:
    """Load the tokenizer with a fallback to "roberta-base"."""
    try:
        tokenizer, _ = load_modified_tokenizer_from_main_config(
            main_config=main_config,
            verbosity=verbosity,
            logger=logger,
        )
    except (
        huggingface_hub.exceptions.ModelHubError,
        FileNotFoundError,
    ):
        logger.exception(
            "Could not load the tokenizer.",
        )
        # Use "roberta-base" as a fallback
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "roberta-base",
        )

    return tokenizer

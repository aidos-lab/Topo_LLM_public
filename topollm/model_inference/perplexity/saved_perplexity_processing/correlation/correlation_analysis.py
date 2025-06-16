# Copyright 2024
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


import logging
import os
import pathlib

import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

from topollm.logging.log_dataframe_info import log_dataframe_info
from topollm.path_management.embeddings.protocol import EmbeddingsPathManager
from topollm.typing.enums import Verbosity

default_logger = logging.getLogger(__name__)


def extract_correlation_columns(
    aligned_df: pd.DataFrame,
    correlation_columns: list[str] | None,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> pd.DataFrame:
    """Extract the columns that are used for the correlation analysis."""
    if correlation_columns is None:
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

    return only_correlation_columns_aligned_df


def compute_and_save_correlation_results_via_mapping_on_all_input_columns(
    only_correlation_columns_df: pd.DataFrame,
    output_directory: os.PathLike,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Calculate and save the correlation results and p-values."""
    methods = {
        "pearson": pearsonr,
        "spearman": spearmanr,
        "kendall": kendalltau,
    }

    for method, func in methods.items():
        # Compute correlation and p-values in a vectorized manner
        corr_df = only_correlation_columns_df.corr(
            method=method,  # type: ignore - corr can accept a callable
        )
        pval_df = only_correlation_columns_df.corr(
            method=lambda x, y, func=func: func(x, y)[1],  # type: ignore - corr can accept a callable
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                f"correlation_results_df using '{method}':\n{corr_df}",  # noqa: G004 - low overhead
            )
            logger.info(
                f"p_value_results_df using '{method}':\n{pval_df}",  # noqa: G004 - low overhead
            )

        # Save correlation and p-value DataFrames to CSV files
        corr_df_save_path = pathlib.Path(
            output_directory,
            f"correlation_results_df_{method}.csv",
        )
        pval_df_save_path = pathlib.Path(
            output_directory,
            f"p_value_results_df_{method}.csv",
        )

        corr_df_save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                f"Saving correlation_results_df to {corr_df_save_path}",  # noqa: G004 - low overhead
            )
        corr_df.to_csv(corr_df_save_path)

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                f"Saving p_value_results_df to {pval_df_save_path}",  # noqa: G004 - low overhead
            )
        pval_df.to_csv(pval_df_save_path)

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                f"Saving correlation_results_df and p_value_results_df using '{method}' to CSV files DONE",  # noqa: G004 - low overhead
            )


def compute_and_save_correlation_results_via_corr_on_all_input_columns(
    only_correlation_columns_df: pd.DataFrame,
    output_directory: os.PathLike,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Calculate and save the correlation results."""
    for method in [
        "pearson",
        "spearman",
        "kendall",
    ]:
        correlation_results_df = only_correlation_columns_df.corr(
            method=method,  # type: ignore - these methods are available
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                f"correlation_results_df using '{method = }':\n{correlation_results_df}",  # noqa: G004 - low overhead
            )

        # Saving correlation_results_df to csv file
        correlation_results_df_save_path = pathlib.Path(
            output_directory,
            f"correlation_via_corr_results_df_{method}.csv",
        )
        correlation_results_df_save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                f"{correlation_results_df_save_path = }",  # noqa: G004 - low overhead
            )
            logger.info(
                f"Saving correlation_results_df using '{method = }' to csv file ...",  # noqa: G004 - low overhead
            )
        correlation_results_df.to_csv(
            path_or_buf=correlation_results_df_save_path,
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                f"Saving correlation_results_df using '{method} = ' to csv file DONE",  # noqa: G004 - low overhead
            )


def compute_and_save_correlation_results_on_all_input_columns_with_embeddings_path_manager(
    only_correlation_columns_df: pd.DataFrame,
    embeddings_path_manager: EmbeddingsPathManager,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Calculate and save the correlation results."""
    for method in [
        "pearson",
        "spearman",
        "kendall",
    ]:
        correlation_results_df = only_correlation_columns_df.corr(
            method=method,  # type: ignore - these methods are available
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                f"Correlation using '{method = }':\n{correlation_results_df}",  # noqa: G004 - low overhead
            )
            logger.info(
                f"{correlation_results_df['local_estimate']['token_log_perplexity'] = }",  # noqa: G004 - low overhead
            )

        # Saving correlation_results_df to csv file
        correlation_results_df_save_path = embeddings_path_manager.get_correlation_results_df_save_path(
            method=method,
        )
        correlation_results_df_save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                f"{correlation_results_df_save_path = }",  # noqa: G004 - low overhead
            )
            logger.info(
                f"Saving correlation_results_df using '{method = }' to csv file ...",  # noqa: G004 - low overhead
            )
        correlation_results_df.to_csv(
            path_or_buf=correlation_results_df_save_path,
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                f"Saving correlation_results_df using '{method} = ' to csv file DONE",  # noqa: G004 - low overhead
            )

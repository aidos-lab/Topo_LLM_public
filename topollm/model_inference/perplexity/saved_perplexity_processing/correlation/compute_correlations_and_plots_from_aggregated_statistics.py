# Copyright 2024
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_1 (author1@example.com)
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
from collections.abc import Callable

import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

from topollm.analysis.correlation.compute_correlations_with_count import (
    IS_SIGNIFICANT_COLUMN_NAME,
    P_VALUE_COLUMN_NAME,
    compute_correlations_with_count,
)
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def compute_correlations_by_dataset(
    df: pd.DataFrame,
    columns_to_correlate: list[str],
    methods: dict[str, Callable],
) -> pd.DataFrame:
    """Compute correlations over all models and checkpoints for each dataset."""
    results = []
    for dataset_name, dataset_group in df.groupby(
        "dataset",
    ):
        correlations_df = compute_correlations_with_count(
            df=dataset_group,
            cols=columns_to_correlate,
            methods=methods,
        )
        correlations_df["dataset"] = dataset_name
        correlations_df["model_without_checkpoint"] = "All Available Models"
        results.append(
            correlations_df,
        )
    return pd.concat(
        results,
        ignore_index=True,
    )


def compute_correlations_by_dataset_and_model(
    df: pd.DataFrame,
    columns_to_correlate: list[str],
    methods: dict[str, Callable],
) -> pd.DataFrame:
    """Compute correlations for each unique combination of dataset and model (without checkpoints)."""
    results = []
    for dataset_name, dataset_group in df.groupby(
        "dataset",
    ):
        for model_name, model_group in dataset_group.groupby(
            "model_without_checkpoint",
        ):
            correlations_df = compute_correlations_with_count(
                df=model_group,
                cols=columns_to_correlate,
                methods=methods,
            )
            correlations_df["dataset"] = dataset_name
            correlations_df["model_without_checkpoint"] = model_name
            results.append(
                correlations_df,
            )
    return pd.concat(
        results,
        ignore_index=True,
    )


def save_correlation_results(
    df: pd.DataFrame,
    file_path: os.PathLike,
    *,
    sort_by_significance: bool = True,
) -> None:
    """Save the correlation results to a CSV file, optionally sorting by significance."""
    if sort_by_significance:
        df = df.sort_values(
            by=[
                IS_SIGNIFICANT_COLUMN_NAME,
                P_VALUE_COLUMN_NAME,
            ],
            ascending=[False, True],
        )

    pathlib.Path(file_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    df.to_csv(
        file_path,
        index=False,
    )


def compute_and_save_correlations_from_aggregated_statistics_df(
    df: pd.DataFrame,
    output_folder: os.PathLike,
    columns_to_correlate: list[str] | None = None,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    # Columns to correlate
    if columns_to_correlate is None:
        columns_to_correlate = [
            "token_perplexity",
            "token_log_perplexity",
            "local_estimate",
        ]

    # Correlation methods to use
    correlation_methods: dict[str, Callable] = {
        "pearson": pearsonr,
        "spearman": spearmanr,
        "kendall": kendalltau,
    }

    # Compute correlations over all models and checkpoints for each dataset
    dataset_level_correlations = compute_correlations_by_dataset(
        df=df,
        columns_to_correlate=columns_to_correlate,
        methods=correlation_methods,
    )

    # Compute correlations for each dataset and model without checkpoints
    model_level_correlations = compute_correlations_by_dataset_and_model(
        df=df,
        columns_to_correlate=columns_to_correlate,
        methods=correlation_methods,
    )

    # Combine both results
    combined_results_df: pd.DataFrame = pd.concat(
        objs=[
            dataset_level_correlations,
            model_level_correlations,
        ],
        ignore_index=True,
    )

    # Save the results to a CSV file
    output_file_path = pathlib.Path(
        output_folder,
        "correlation_significance_sorted.csv",
    )

    save_correlation_results(
        df=combined_results_df,
        file_path=output_file_path,
    )

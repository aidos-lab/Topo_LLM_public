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
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def compute_row_correlations(
    array: np.ndarray,
    method: str = "pearson",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the correlation coefficients between the rows of a numpy array.

    Args:
    ----
        array:
            The input array with shape (num_samples, num_points).
        method:
            The correlation method to use - "pearson", "kendall", or "spearman".

    Returns:
    -------
        Tuple[np.ndarray, np.ndarray]:
            A tuple containing the correlation matrix of shape (num_samples, num_samples)
            and the p-value matrix of shape (num_samples, num_samples).

    """
    num_samples: int = array.shape[0]
    correlation_matrix = np.zeros(
        shape=(num_samples, num_samples),
    )
    p_matrix = np.zeros(
        shape=(num_samples, num_samples),
    )

    for i in range(num_samples):
        for j in range(num_samples):
            if method == "pearson":
                correlation_matrix[i, j], p_matrix[i, j] = pearsonr(array[i], array[j])
            elif method == "kendall":
                correlation_matrix[i, j], p_matrix[i, j] = kendalltau(array[i], array[j])
            elif method == "spearman":
                correlation_matrix[i, j], p_matrix[i, j] = spearmanr(array[i], array[j])
            else:
                msg: str = f"Unsupported correlation method: {method = }"
                raise ValueError(msg)

    return correlation_matrix, p_matrix


def compute_and_save_correlations(
    arrays_truncated_stacked: np.ndarray,
    results_base_directory: pathlib.Path,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Compute and save the correlation coefficients for the given arrays."""
    methods: list[str] = [
        "pearson",
        "kendall",
        "spearman",
    ]
    # # # #
    # Compute and save the correlation coefficients for each method
    for method in methods:
        correlation_save_path: pathlib.Path = results_base_directory / f"correlation_coefficients_{method}.csv"
        p_value_save_path: pathlib.Path = results_base_directory / f"correlation_p_values_{method}.csv"
        correlation_matrix, p_matrix = compute_row_correlations(
            array=arrays_truncated_stacked,
            method=method,
        )
        pd.DataFrame(
            data=correlation_matrix,
        ).to_csv(
            path_or_buf=correlation_save_path,
            index=False,
        )
        pd.DataFrame(
            data=p_matrix,
        ).to_csv(
            path_or_buf=p_value_save_path,
            index=False,
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"{method.capitalize()} correlation coefficients saved to: {correlation_save_path = }",  # noqa: G004 - low overhead
            )
            logger.info(
                msg=f"{method.capitalize()} p-values saved to: {p_value_save_path = }",  # noqa: G004 - low overhead
            )

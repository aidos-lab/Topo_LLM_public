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

"""Functions for computing correlations between columns in a DataFrame."""

import logging
from collections.abc import Callable

import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

P_VALUE_COLUMN_NAME = "p_value"
IS_SIGNIFICANT_COLUMN_NAME = "is_significant"

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def compute_correlations_with_count(
    df: pd.DataFrame,
    cols: list[str],
    methods: dict[str, Callable] | None = None,
    significance_level: float = 0.05,
) -> pd.DataFrame:
    """Compute correlations and include the count of data points used.

    If the number of data points is less than 2, the correlation is not computed.
    If methods is None, the default correlation methods are used.
    """
    # Set default correlation methods
    if methods is None:
        methods = {
            "pearson": pearsonr,
            "spearman": spearmanr,
            "kendall": kendalltau,
        }

    results: list[dict] = []
    for method_name, method_func in methods.items():
        for i in range(
            len(cols),
        ):
            for j in range(
                i + 1,
                len(cols),
            ):
                (
                    col1,
                    col2,
                ) = cols[i], cols[j]
                valid_data: pd.DataFrame = df[[col1, col2]].dropna()
                n: int = len(valid_data)
                if n >= 2:  # Ensure sufficient data points  # noqa: PLR2004 - This is a valid check
                    correlation, p_value = method_func(
                        valid_data[col1],
                        valid_data[col2],
                    )
                    results.append(
                        {
                            "method": method_name,
                            "column_1": col1,
                            "column_2": col2,
                            "correlation": correlation,
                            P_VALUE_COLUMN_NAME: p_value,
                            IS_SIGNIFICANT_COLUMN_NAME: p_value < significance_level,
                            "n": n,
                        },
                    )
    results_df = pd.DataFrame(
        data=results,
    )

    return results_df

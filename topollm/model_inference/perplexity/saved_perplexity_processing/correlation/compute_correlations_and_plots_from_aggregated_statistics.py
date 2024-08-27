# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (ruppik@hhu.de)
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

from collections.abc import Callable

import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr


# Function to load the dataset from a CSV file
def load_dataset(
    file_path: str,
) -> pd.DataFrame:
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)


# Function to compute correlations with significance and count of data points
def compute_correlations_with_count(
    df: pd.DataFrame,
    cols: list[str],
    methods: dict[str, Callable],
    significance_level: float = 0.05,
) -> pd.DataFrame:
    """Compute correlations and include the count of data points used."""
    results = []
    for method_name, method_func in methods.items():
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                col1, col2 = cols[i], cols[j]
                valid_data = df[[col1, col2]].dropna()
                n = len(valid_data)
                if n >= 2:  # Ensure sufficient data points
                    correlation, p_value = method_func(valid_data[col1], valid_data[col2])
                    results.append(
                        {
                            "Method": method_name,
                            "Variable 1": col1,
                            "Variable 2": col2,
                            "Correlation": correlation,
                            "P-Value": p_value,
                            "Significant": p_value < significance_level,
                            "N": n,
                        },
                    )
    return pd.DataFrame(results)


# Function to compute correlations for each dataset
def compute_correlations_by_dataset(
    df: pd.DataFrame,
    columns_to_correlate: list[str],
    methods: dict[str, Callable],
) -> pd.DataFrame:
    """Compute correlations over all models and checkpoints for each dataset."""
    results = []
    for dataset_name, dataset_group in df.groupby("dataset"):
        correlations_df = compute_correlations_with_count(dataset_group, columns_to_correlate, methods)
        correlations_df["Dataset"] = dataset_name
        correlations_df["Model"] = "All Models"
        results.append(correlations_df)
    return pd.concat(results, ignore_index=True)


# Function to compute correlations by dataset and model (excluding checkpoints)
def compute_correlations_by_dataset_and_model(
    df: pd.DataFrame,
    columns_to_correlate: list[str],
    methods: dict[str, Callable],
) -> pd.DataFrame:
    """Compute correlations for each unique combination of dataset and model (without checkpoints)."""
    results = []
    for dataset_name, dataset_group in df.groupby("dataset"):
        for model_name, model_group in dataset_group.groupby("model_without_checkpoint"):
            correlations_df = compute_correlations_with_count(model_group, columns_to_correlate, methods)
            correlations_df["Dataset"] = dataset_name
            correlations_df["Model"] = model_name
            results.append(correlations_df)
    return pd.concat(results, ignore_index=True)


# Function to save the correlation results to a CSV file
def save_correlation_results(
    df: pd.DataFrame,
    file_path: str,
    *,
    sort_by_significance: bool = True,
) -> None:
    """Save the correlation results to a CSV file, optionally sorting by significance."""
    if sort_by_significance:
        df = df.sort_values(
            by=["Significant", "P-Value"],
            ascending=[False, True],
        )
    df.to_csv(
        file_path,
        index=False,
    )


def main() -> None:
    # TODO: Update the file paths as needed
    # TODO: Call this function from the main entry point

    # File paths
    input_file_path = "/mnt/data/aggregated_statistics.csv"
    output_file_path = "/mnt/data/correlation_significance_sorted_with_counts.csv"

    # Load the dataset
    df = load_dataset(input_file_path)

    # Columns to correlate
    columns_to_correlate = ["token_perplexity", "token_log_perplexity", "local_estimate"]

    # Correlation methods to use
    correlation_methods = {"pearson": pearsonr, "spearman": spearmanr, "kendall": kendalltau}

    # Step 1: Compute correlations over all models and checkpoints for each dataset
    dataset_level_correlations = compute_correlations_by_dataset(df, columns_to_correlate, correlation_methods)

    # Step 2: Compute correlations for each dataset and model without checkpoints
    model_level_correlations = compute_correlations_by_dataset_and_model(df, columns_to_correlate, correlation_methods)

    # Combine both results
    combined_results_df = pd.concat([dataset_level_correlations, model_level_correlations], ignore_index=True)

    # Step 3: Save the results to a CSV file, sorted by significance
    save_correlation_results(combined_results_df, output_file_path)

    print(f"Correlation results saved to {output_file_path}")

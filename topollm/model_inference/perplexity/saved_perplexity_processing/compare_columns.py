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

import pandas as pd


def compare_columns(
    df1: pd.DataFrame,
    col1: str,
    df2: pd.DataFrame,
    col2: str,
) -> pd.DataFrame:
    """Compare two columns from different DataFrames and return a DataFrame highlighting the differences.

    Args:
    ----
        df1 (pd.DataFrame): First DataFrame.
        col1 (str): Column name in the first DataFrame.
        df2 (pd.DataFrame): Second DataFrame.
        col2 (str): Column name in the second DataFrame.

    Returns:
    -------
        pd.DataFrame: DataFrame containing the rows where the columns differ.

    """
    # Reset indices to ensure comparison by position
    df1_reset = df1[col1].reset_index(
        drop=True,
    )
    df2_reset = df2[col2].reset_index(
        drop=True,
    )

    # Ensure the columns have the same length
    if len(df1_reset) != len(df2_reset):
        msg = "The columns must have the same length to compare."
        raise ValueError(msg)

    # Create a DataFrame to compare the two columns
    comparison_df = pd.DataFrame(
        {
            f"{col1}_df1": df1_reset,
            f"{col2}_df2": df2_reset,
        },
    )

    # Add a column to indicate where the values are not equal
    comparison_df["Equal"] = comparison_df[f"{col1}_df1"] == comparison_df[f"{col2}_df2"]

    # Filter rows where the values are not equal
    discrepancies = comparison_df[~comparison_df["Equal"]]

    return discrepancies

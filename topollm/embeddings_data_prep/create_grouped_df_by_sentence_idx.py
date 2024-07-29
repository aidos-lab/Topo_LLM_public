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

import pandas as pd


def create_grouped_df_by_sentence_idx(
    full_df: pd.DataFrame,
    meta_tokens_column_name: str = "meta_tokens",
) -> pd.DataFrame:
    """Group the DataFrame by 'sentence_idx', aggregate tokens into lists and concatenate them."""
    # Group by 'sentence_idx' and aggregate tokens into lists
    grouped_df: pd.DataFrame = (
        full_df.groupby(
            "sentence_idx",
            sort=False,
        )
        .agg(
            tokens_list=(meta_tokens_column_name, list),
        )
        .reset_index()
    )

    # Add an additional column with concatenated tokens
    grouped_df["concatenated_tokens"] = grouped_df["tokens_list"].apply(
        " ".join,
    )

    return grouped_df

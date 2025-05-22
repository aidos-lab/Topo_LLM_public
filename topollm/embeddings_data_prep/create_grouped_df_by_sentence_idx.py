# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Matthias Ruppik (mail@ruppik.net)
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

"""Grouping data frame by sentence index."""

import pandas as pd

from topollm.config_classes.data_processing_column_names.data_processing_column_names import DataProcessingColumnNames

default_data_processing_column_names = DataProcessingColumnNames()


def create_grouped_df_by_sentence_idx(
    full_df: pd.DataFrame,
    data_processing_column_names: DataProcessingColumnNames = default_data_processing_column_names,
) -> pd.DataFrame:
    """Group the DataFrame by the sentence index, aggregate tokens into lists and concatenate them."""
    # Group by sentence index and aggregate tokens into lists
    grouped_df: pd.DataFrame = (
        full_df.groupby(
            data_processing_column_names.sentence_idx,
            sort=False,
        )
        .agg(
            tokens_list=(data_processing_column_names.token_name, list),
        )
        .reset_index()
    )

    # Add an additional column with concatenated tokens
    grouped_df[data_processing_column_names.concatenated_tokens] = grouped_df[
        data_processing_column_names.tokens_list
    ].apply(
        " ".join,
    )

    return grouped_df

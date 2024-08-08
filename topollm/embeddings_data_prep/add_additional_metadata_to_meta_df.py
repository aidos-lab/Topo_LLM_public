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

"""Add additional metadata to the metadata DataFrame."""

import pandas as pd

from topollm.config_classes.data_processing_column_names.data_processing_column_names import DataProcessingColumnNames
from topollm.embeddings_data_prep.create_grouped_df_by_sentence_idx import create_grouped_df_by_sentence_idx
from topollm.typing.types import TransformersTokenizer

default_data_processing_column_names = DataProcessingColumnNames()


def add_token_name_column_to_meta_frame(
    input_df: pd.DataFrame,
    tokenizer: TransformersTokenizer,
    data_processing_column_names: DataProcessingColumnNames = default_data_processing_column_names,
) -> pd.DataFrame:
    """Add a column with the meta tokens to the metadata DataFrame."""
    # Check that the input DataFrame has the necessary columns
    if data_processing_column_names.token_id not in input_df.columns:
        msg = f"The input DataFrame must have a column named {data_processing_column_names.token_id = }."
        raise ValueError(msg)

    # x of type 'numpy.int64' needs to be explicitly converted to an integer,
    # otherwise the convert_ids_to_tokens() method will raise the error:
    # TypeError: 'numpy.int64' object is not iterable
    token_names_list = [
        tokenizer.convert_ids_to_tokens(int(x))
        for x in list(
            input_df[data_processing_column_names.token_id],
        )
    ]

    # Add the decoded tokens to the DataFrame
    input_df[data_processing_column_names.token_name] = token_names_list

    return input_df


def add_additional_metadata_to_meta_df(
    full_df: pd.DataFrame,
    meta_df_to_modify: pd.DataFrame,
    tokenizer: TransformersTokenizer,
    data_processing_column_names: DataProcessingColumnNames = default_data_processing_column_names,
    *,
    write_tokens_list_to_meta: bool = True,
    write_concatenated_tokens_to_meta: bool = True,
) -> pd.DataFrame:
    """Add additional metadata to the metadata DataFrame."""
    full_with_token_name_df: pd.DataFrame = add_token_name_column_to_meta_frame(
        input_df=full_df,
        tokenizer=tokenizer,
        data_processing_column_names=data_processing_column_names,
    )

    grouped_df: pd.DataFrame = create_grouped_df_by_sentence_idx(
        full_df=full_with_token_name_df,
        data_processing_column_names=data_processing_column_names,
    )

    meta_df_to_modify = meta_df_to_modify.merge(
        right=grouped_df,
        on=data_processing_column_names.sentence_idx,
    )

    # Remove columns if they should not be saved
    if not write_tokens_list_to_meta:
        meta_df_to_modify = meta_df_to_modify.drop(
            columns=[data_processing_column_names.tokens_list],
        )
    if not write_concatenated_tokens_to_meta:
        meta_df_to_modify = meta_df_to_modify.drop(
            columns=[data_processing_column_names.concatenated_tokens],
        )

    return meta_df_to_modify

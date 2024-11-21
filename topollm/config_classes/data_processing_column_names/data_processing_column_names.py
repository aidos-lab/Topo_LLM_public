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

"""Column names for data processing."""

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel


class DataProcessingColumnNames(ConfigBaseModel):
    """Column names for data processing."""

    concatenated_tokens: str = Field(
        default="concatenated_tokens",
        title="Column name for the concatenated tokens.",
        description="The column name for the concatenated tokens.",
    )

    embedding_vectors: str = Field(
        default="embedding_vectors",
        title="Column name for the embedding vectors.",
        description="The column name for the embedding vectors.",
    )

    sentence_idx: str = Field(
        default="sentence_idx",
        title="Column name for the sentence index.",
        description="The column name for the sentence index.",
    )

    subsample_idx: str = Field(
        default="subsample_idx",
        title="Column name for the subsample index.",
        description="The column name for the subsample index.",
    )

    token_id: str = Field(
        default="token_id",
        title="Column name for the token_id, i.e., the number produced by the tokenizer.",
        description="The column name for the token_id.",
    )

    tokens_list: str = Field(
        default="tokens_list",
        title="Column name for the list of tokens making up a sentence.",
        description="The column name for the list of tokens making up a sentence.",
    )

    token_name: str = Field(
        default="token_name",
        title="Column name for the decoded token_id information.",
        description="The column name for the decoded token_id information.",
    )

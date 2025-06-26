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

    input_ids: str = Field(
        default="input_ids",
        title="Column name for the input_ids, i.e., the number produced by the tokenizer.",
        description="The column name for the input_ids.",
    )

    tokens_list: str = Field(
        default="tokens_list",
        title="Column name for the list of tokens making up a sentence.",
        description="The column name for the list of tokens making up a sentence.",
    )

    token_name: str = Field(
        default="token_name",
        title="Column name for the decoded input_ids information.",
        description="The column name for the decoded input_ids information.",
    )

    pos_tags_name: str = Field(
        default="POS",
        title="Column name for the part-of-speech tags.",
        description="The column name for the part-of-speech tags.",
    )

    bio_tags_name: str = Field(
        default="bio_tag",
        title="Column name for the BIO tags.",
        description="The column name for the BIO tags.",
    )

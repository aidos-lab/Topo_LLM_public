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


import numpy as np
import pandas as pd


def add_token_log_perplexity_column(
    token_perplexities_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add a column with the log perplexity to the DataFrame."""
    token_perplexities_df["token_log_perplexity"] = token_perplexities_df["token_perplexity"].apply(
        lambda x: np.log(x),
    )
    # Replace `-inf` values with `0.0`
    token_perplexities_df["token_log_perplexity"] = token_perplexities_df["token_log_perplexity"].replace(
        -np.inf,
        0.0,
    )

    return token_perplexities_df

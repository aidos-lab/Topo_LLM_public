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

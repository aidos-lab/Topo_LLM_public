"""Function to summarize a value for logging purposes."""

from typing import Any

import numpy as np
import pandas as pd


def summarize_value(
    value: Any,
    key: str | None = None,
    fallback_truncation_length: int = 100,
) -> str:
    """Summarize a value for logging purposes.

    This function provides a concise summary depending on the type of `value`.

    Args:
        value:
            The value to summarize.
        key:
            The key to use to describe the value.
        fallback_truncation_length:
            The maximum length of the string representation of the value if no other summary is available.

    Returns:
        str: A summary string describing the value.

    """
    key_str: str = f"{key = }:\n" if key is not None else ""

    if value is None:
        value_str: str = "\t\tNone"
    elif isinstance(
        value,
        np.ndarray,
    ):
        value_str = f"\t\tNumPy array with {value.shape = } and {value.dtype = }."

        # If the array is one-dimensional, also compute the mean and standard deviation
        if len(value.shape) == 1:
            value_str += f"\n\t\tnp.mean: {np.mean(a=value):.3f}; np.std(ddof=1): {np.std(a=value, ddof=1):.3f}"
    elif isinstance(
        value,
        pd.DataFrame,
    ):
        value_str = f"\t\tDataFrame with {value.shape = } and columns {list(value.columns)}"
    elif isinstance(
        value,
        dict,
    ):
        value_str = f"\t\tdict with keys {list(value.keys())}"
    else:
        value_str = str(object=value)[:fallback_truncation_length]

    result: str = key_str + value_str

    return result

# Copyright 2024-2025
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

"""Utility functions for handling dictionaries."""

import logging
import pathlib
from typing import Any

from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def flatten_dict(
    d: dict[str, Any],
    separator: str = "_",
    parent_key: str = "",
) -> dict[str, Any]:
    """Flattens a nested dictionary by concatenating keys with a separator.

    Args:
        d: The dictionary to flatten.
        separator: The character used to join nested keys.
        parent_key: The base key for recursion (used internally).

    Returns:
        A flattened dictionary.

    """
    items: dict[str, Any] = {}
    for key, value in d.items():
        new_key: str = f"{parent_key}{separator}{key}" if parent_key else key
        if isinstance(
            value,
            dict,
        ):
            items.update(
                flatten_dict(
                    d=value,
                    separator=separator,
                    parent_key=new_key,
                ),
            )
        else:
            items[new_key] = value
    return items


def filter_list_of_dictionaries_by_key_value_pairs(  # noqa: D417 - we do not add the verbosity and logger argument to the docstring
    list_of_dicts: list[dict],
    key_value_pairs: dict,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> list[dict]:
    """Filter a list of dictionaries by a set of key-value pairs.

    Args:
        list_of_dicts: The list of dictionaries to filter.
        key_value_pairs: The key-value pairs to filter by.

    Returns:
        The filtered list of dictionaries.

    """
    filtered_list_of_dicts: list[dict] = [
        single_dict
        for single_dict in list_of_dicts
        if all(single_dict[key] == value for key, value in key_value_pairs.items())
    ]

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{len(list_of_dicts) = } filtered to {len(filtered_list_of_dicts) = }.",  # noqa: G004 - low overhead
        )

    return filtered_list_of_dicts


def dictionary_to_partial_path(
    dictionary: dict,
    key_value_separator: str = "=",
) -> pathlib.Path:
    """Convert a dictionary to a partial path.

    Args:
        dictionary: The dictionary to convert.
        key_value_separator: The separator between key and value.

    Returns:
        The partial path.

    """
    partial_path_list: list[str] = [f"{key}{key_value_separator}{value}" for key, value in dictionary.items()]

    # Unpack the list into the Path constructor
    result = pathlib.Path(
        *partial_path_list,
    )

    return result


def generate_fixed_parameters_text_from_dict(
    filters_dict: dict[str, Any],
) -> str:
    """Generate a string representation of the fixed parameters used for filtering.

    Args:
        filters_dict:
            A dictionary of column names and corresponding values used for filtering.

    Returns:
        str:
            A formatted string suitable for display in the plot.

    """
    return "\n".join([f"{key}: {value}" for key, value in filters_dict.items()])

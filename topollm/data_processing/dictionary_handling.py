# Copyright 2024-2025
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (mail@ruppik.net)
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

"""Utility functions for handling dictionaries."""

from typing import Any


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

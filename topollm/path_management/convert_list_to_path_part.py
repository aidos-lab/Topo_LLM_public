# Copyright 2023-2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (ruppik@hhu.de)
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

import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    # Fallback to the strenum package for Python 3.10 and below.
    # Run `python3 -m pip install strenum` for python < 3.11
    from strenum import StrEnum


def convert_list_to_path_part(
    input_list: list,
    delimiter: str = "_",
) -> str:
    """Convert a list to a string suitable for file paths.

    Args:
    ----
        input_list:
            List which should be converted to a string.
        delimiter:
            Delimiter to join the string values.

    Returns:
    -------
        str: A string suitable for file paths.

    """
    # Convert the elements of the list to strings
    input_list_with_string_values = [str(item) for item in input_list]
    # Join the string values with the specified delimiter
    path_part = delimiter.join(
        input_list_with_string_values,
    )
    return path_part


def test_convert_list_to_path_part() -> None:
    """Example usage of convert_str_enum_list_to_path_part function."""
    # TODO(Ben): Move this to the tests directory
    list_of_examples: list[list] = [
        [],
        [
            "encoder.layer.0.",
            "encoder.layer.1.",
            "encoder.layer.2.",
            "encoder.layer.3.",
            "encoder.layer.4.",
            "encoder.layer.5.",
        ],
    ]

    for example in list_of_examples:
        path_part = convert_list_to_path_part(
            input_list=example,
        )
        print(path_part)


test_convert_list_to_path_part()

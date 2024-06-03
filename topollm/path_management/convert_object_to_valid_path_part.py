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
import warnings

from topollm.path_management.validate_path_part import validate_path_part

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

    if not validate_path_part(
        path_part=path_part,
    ):
        warnings.warn(
            f"{path_part = } is not suitable for file paths.",
            stacklevel=1,
        )

    return path_part

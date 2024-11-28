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


from topollm.path_management.truncate_length_of_desc import truncate_length_of_desc


def sanitize_dirname(
    dir_name: str,
) -> str:
    """Sanitizes a directory name by replacing all slashes with underscores.

    Args:
    ----
        dir_name: The directory name to sanitize.

    Returns:
    -------
        The sanitized directory name.

    """
    result = dir_name.replace(
        "/",
        "_",
    ).replace(
        "\\",
        "_",
    )

    result: str = truncate_length_of_desc(
        desc=result,
    )

    if len(result) == 0:
        result = "no_overrides"

    return result

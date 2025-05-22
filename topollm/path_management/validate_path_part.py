# Copyright 2023-2024
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_1 (author1@example.com)
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


def validate_path_part(
    path_part: str,
) -> bool:
    """Validate if a string is suitable for file paths.

    This also checks for common issues that would appear when using gsutil for Google Cloud bucket operations,
    in particular, the following characters are not allowed because they might be interpreted as wildcard:
    `*`, `?`, `[`, `]`.
    Also see the following discussions:
    - https://stackoverflow.com/questions/42087510/gsutil-ls-returns-error-contains-wildcard
    - https://github.com/GoogleCloudPlatform/gsutil/issues/290
    - https://cloud.google.com/storage/docs/gsutil/addlhelp/WildcardNames
    """
    # Check if the path part is a string
    if not isinstance(
        path_part,
        str,
    ):
        return False

    gsutil_wildcard_characters: list[str] = [
        "*",
        "?",
        "[",
        "]",
    ]

    osx_forbidden_characters: list[str] = [
        ":",
    ]

    windows_forbidden_characters: list[str] = [
        "<",
        ">",
        ":",
        '"',
        "|",
        "?",
        "*",
    ]

    characters_to_avoid: list[str] = (
        gsutil_wildcard_characters + osx_forbidden_characters + windows_forbidden_characters
    )

    # Check if the path part does not contain any of the following characters
    if any(char in path_part for char in characters_to_avoid):
        return False
    return True

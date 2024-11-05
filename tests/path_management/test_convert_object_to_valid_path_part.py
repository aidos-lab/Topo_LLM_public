# Copyright 2023-2024
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

"""Tests for the functions which convert objects into valid path parts."""

import logging

from topollm.path_management.convert_object_to_valid_path_part import convert_list_to_path_part
from topollm.path_management.validate_path_part import validate_path_part


def test_convert_list_to_path_part(
    logger_fixture: logging.Logger,
) -> None:
    """Example usage of convert_str_enum_list_to_path_part function."""
    list_of_examples: list[list] = [
        [],  # empty list
        [
            "",
        ],  # list with one empty string
        [
            "encoder.layer.0.",
            "encoder.layer.1.",
            "encoder.layer.2.",
            "encoder.layer.3.",
            "encoder.layer.4.",
            "encoder.layer.5.",
        ],  # list with multiple strings
        [
            "query",
            "key",
            "value",
        ],  # list with multiple strings
        [
            -1,
            -2,
            -3,
        ],  # list with multiple integers
    ]

    for example in list_of_examples:
        path_part = convert_list_to_path_part(
            input_list=example,
        )
        logger_fixture.info(
            "path_part:\n%s",
            path_part,
        )

        assert validate_path_part(  # noqa: S101 - pytest assertion
            path_part=path_part,
        )

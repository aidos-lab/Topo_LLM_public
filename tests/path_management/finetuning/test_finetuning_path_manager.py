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

"""Tests for the finetuning path manager."""

import logging
import pathlib

from topollm.path_management.finetuning import protocol
from topollm.path_management.validate_path_part import validate_path_part


class TestFinetuningPathManager:
    """Tests for the finetuning path manager."""

    def test_finetuned_model_dir(
        self,
        finetuning_path_manager_basic: protocol.FinetuningPathManager,
        logger_fixture: logging.Logger,
    ) -> None:
        result: pathlib.Path = finetuning_path_manager_basic.finetuned_model_dir
        logger_fixture.info(
            "finetuned_model_dir:\n%s",
            result,
        )

        assert isinstance(  # noqa: S101 - pytest assertion
            result,
            pathlib.Path,
        )

        assert validate_path_part(  # noqa: S101 - pytest assertion
            path_part=str(object=result),
        )

    def test_logging_dir(
        self,
        finetuning_path_manager_basic: protocol.FinetuningPathManager,
        logger_fixture: logging.Logger,
    ) -> None:
        result: pathlib.Path | None = finetuning_path_manager_basic.logging_dir
        logger_fixture.info(
            "logging_dir:\n%s",
            result,
        )

        assert isinstance(  # noqa: S101 - pytest assertion
            result,
            pathlib.Path | None,
        )

        if result is not None:
            assert validate_path_part(  # noqa: S101 - pytest assertion
                path_part=str(object=result),
            )

    def test_get_finetuned_short_model_name(
        self,
        finetuning_path_manager_basic: protocol.FinetuningPathManager,
        logger_fixture: logging.Logger,
    ) -> None:
        result: str = finetuning_path_manager_basic.get_finetuned_short_model_name()

        logger_fixture.info(
            "finetuned_short_model_name:\n%s",
            result,
        )

        assert isinstance(  # noqa: S101 - pytest assertion
            result,
            str,
        )

    def test_get_finetuning_parameters_description(
        self,
        finetuning_path_manager_basic: protocol.FinetuningPathManager,
        logger_fixture: logging.Logger,
    ) -> None:
        result: str = finetuning_path_manager_basic.get_finetuning_parameters_description_for_short_model_name()
        logger_fixture.info(
            "finetuning_parameters_description:\n%s",
            result,
        )

        assert isinstance(  # noqa: S101 - pytest assertion
            result,
            str,
        )

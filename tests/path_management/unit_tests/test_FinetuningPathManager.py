# coding=utf-8
#
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

import logging
import pathlib

import pytest

from topollm.path_management.finetuning import FinetuningPathManagerProtocol


@pytest.fixture
def finetuning_path_manager(
    request: pytest.FixtureRequest,
) -> FinetuningPathManagerProtocol.FinetuningPathManager:
    # This uses the request fixture to dynamically get a fixture by name.
    return request.getfixturevalue(
        argname=request.param,
    )


@pytest.mark.parametrize(
    "finetuning_path_manager",
    [
        "finetuning_path_manager_basic",
    ],
    indirect=True,
)
class TestEmbeddingsPathManager:
    def test_finetuned_model_dir(
        self,
        finetuning_path_manager: FinetuningPathManagerProtocol.FinetuningPathManager,
        logger_fixture: logging.Logger,
    ) -> None:
        result = finetuning_path_manager.finetuned_model_dir
        logger_fixture.info(f"finetuned_model_dir:\n" f"{result = }")

        assert isinstance(
            result,
            pathlib.Path,
        )

        return None

    def test_logging_dir(
        self,
        finetuning_path_manager: FinetuningPathManagerProtocol.FinetuningPathManager,
        logger_fixture: logging.Logger,
    ) -> None:
        result = finetuning_path_manager.logging_dir
        logger_fixture.info(f"logging_dir:\n" f"{result = }")

        assert isinstance(
            result,
            pathlib.Path | None,
        )

        return None

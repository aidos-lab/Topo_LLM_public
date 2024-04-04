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

from topollm.path_management.embeddings import EmbeddingsPathManagerProtocol


@pytest.fixture
def embeddings_path_manager(
    request: pytest.FixtureRequest,
) -> EmbeddingsPathManagerProtocol.EmbeddingsPathManager:
    # This uses the request fixture to dynamically get a fixture by name.
    return request.getfixturevalue(
        argname=request.param,
    )


@pytest.mark.parametrize(
    "embeddings_path_manager",
    [
        "embeddings_path_manager_separate_directories",
    ],
    indirect=True,
)
class TestEmbeddingsPathManager:
    def test_data_dir(
        self,
        embeddings_path_manager: EmbeddingsPathManagerProtocol.EmbeddingsPathManager,
        logger_fixture: logging.Logger,
    ) -> None:
        result = embeddings_path_manager.data_dir
        logger_fixture.info(f"data_dir:\n" f"{result = }")

        assert isinstance(
            result,
            pathlib.Path,
        )

        return None

    def test_array_dir_absolute_path(
        self,
        embeddings_path_manager: EmbeddingsPathManagerProtocol.EmbeddingsPathManager,
        logger_fixture: logging.Logger,
    ) -> None:
        result = embeddings_path_manager.array_dir_absolute_path
        logger_fixture.info(f"array_dir_absolute_path:\n" f"{result = }")

        assert isinstance(
            result,
            pathlib.Path,
        )

        return None

    def test_metadata_dir_absolute_path(
        self,
        embeddings_path_manager: EmbeddingsPathManagerProtocol.EmbeddingsPathManager,
        logger_fixture: logging.Logger,
    ) -> None:
        result = embeddings_path_manager.metadata_dir_absolute_path
        logger_fixture.info(f"metadata_dir_absolute_path:\n" f"{result = }")

        assert isinstance(
            result,
            pathlib.Path,
        )

        return None

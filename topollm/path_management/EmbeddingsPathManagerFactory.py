# coding=utf-8
#
# Copyright 2024
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

import logging

from topollm.config_classes.MainConfig import MainConfig
from topollm.path_management.EmbeddingsPathManagerProtocol import (
    EmbeddingsPathManager,
)
from topollm.path_management.EmbeddingsPathManagerSeparateDirectories import (
    EmbeddingsPathManagerSeparateDirectories,
)


def get_embeddings_path_manager(
    config: MainConfig,
    logger: logging.Logger = logging.getLogger(__name__),
) -> EmbeddingsPathManager:
    path_manger = EmbeddingsPathManagerSeparateDirectories(
        data_config=config.data,
        embeddings_config=config.embeddings,
        paths_config=config.paths,
        transformations_config=config.transformations,
        verbosity=config.verbosity,
        logger=logger,
    )

    return path_manger

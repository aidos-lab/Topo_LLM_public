# Copyright 2024
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

"""Factory for creating a finetuning path manager."""

import logging

from topollm.config_classes.main_config import MainConfig
from topollm.path_management.finetuning.finetuning_path_manager_basic import (
    FinetuningPathManagerBasic,
)
from topollm.path_management.finetuning.protocol import (
    FinetuningPathManager,
)

default_logger = logging.getLogger(__name__)


def get_finetuning_path_manager(
    main_config: MainConfig,
    logger: logging.Logger = default_logger,
) -> FinetuningPathManager:
    """Get a finetuning path manager based on the main configuration."""
    path_manger = FinetuningPathManagerBasic(
        data_config=main_config.data,
        paths_config=main_config.paths,
        finetuning_config=main_config.finetuning,
        verbosity=main_config.verbosity,
        logger=logger,
    )

    return path_manger

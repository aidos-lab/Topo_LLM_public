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

from topollm.config_classes.data.DataConfig import DataConfig
from topollm.config_classes.finetuning.FinetuningConfig import FinetuningConfig
from topollm.config_classes.MainConfig import MainConfig
from topollm.config_classes.PathsConfig import PathsConfig
from topollm.path_management.finetuning.FinetuningPathManagerBasic import (
    FinetuningPathManagerBasic,
)
from topollm.path_management.finetuning.FinetuningPathManagerProtocol import (
    FinetuningPathManager,
)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Globals

# END Globals
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def get_finetuning_path_manager(
    main_config: MainConfig,
    logger: logging.Logger = logging.getLogger(__name__),
) -> FinetuningPathManager:
    path_manger = FinetuningPathManagerBasic(
        data_config=main_config.data,
        paths_config=main_config.paths,
        finetuning_config=main_config.finetuning,
        verbosity=main_config.verbosity,
        logger=logger,
    )

    return path_manger

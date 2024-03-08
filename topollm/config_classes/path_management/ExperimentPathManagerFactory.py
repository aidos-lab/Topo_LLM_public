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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Imports

# System imports
import logging

# External imports

# Local imports
from convlab.tda.config_classes.ExperimentConfig import ExperimentConfig
from convlab.tda.config_classes.path_management.BioTaggerExperimentPathManager import (
    BioTaggerExperimentPathManager,
)
from convlab.tda.config_classes.path_management.ExperimentPathManagerProtocol import (
    ExperimentPathManagerProtocol,
)

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def get_experiment_path_manager(
    config: ExperimentConfig,
    logger: logging.Logger = logging.getLogger(__name__),
) -> ExperimentPathManagerProtocol:
    experiment_path_manger = BioTaggerExperimentPathManager(
        config=config,
        logger=logger,
    )

    return experiment_path_manger

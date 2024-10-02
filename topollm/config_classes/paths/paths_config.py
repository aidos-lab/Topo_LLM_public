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

"""Configuration class for specifying paths."""

import pathlib

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel
from topollm.config_classes.constants import TOPO_LLM_REPOSITORY_BASE_PATH


class PathsConfig(ConfigBaseModel):
    """Configurations for specifying paths."""

    data_dir: pathlib.Path = Field(
        default=pathlib.Path(
            TOPO_LLM_REPOSITORY_BASE_PATH,
            "data",
        ),
        title="Data path.",
        description="The path to the data.",
    )

    repository_base_path: pathlib.Path = Field(
        default=TOPO_LLM_REPOSITORY_BASE_PATH,
        title="Repository base path.",
        description="The base path to the repository.",
    )

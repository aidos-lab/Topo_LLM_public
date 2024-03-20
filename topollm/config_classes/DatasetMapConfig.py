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

# Local imports
from pydantic import Field

# Third-party imports
from topollm.config_classes.ConfigBaseModel import ConfigBaseModel


# END Globals
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class DatasetMapConfig(ConfigBaseModel):
    """
    Configurations for specifying dataset map.
    """

    batch_size: int = Field(
        ...,
        title="Batch size for mapping tokenization on dataset.",
        description="The batch size for mapping tokenization on dataset.",
    )

    num_proc: int = Field(
        ...,
        title="Number of processes for mapping tokenization on dataset.",
        description="The number of processes for mapping tokenization on dataset.",
    )
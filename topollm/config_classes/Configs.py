# coding=utf-8
#
# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Julius von Rohrscheidt (julius.rohrscheidt@helmholtz-muenchen.de)
# Benjamin Ruppik (ruppik@hhu.de)
#
# This code was generated with the help of AI writing assistants
# including GitHub Copilot, ChatGPT, Bing Chat.
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

# Standard library imports
import json
from os import PathLike
import pathlib
import pprint
from abc import ABC, abstractmethod
from typing import IO

# Third party imports
from pydantic import BaseModel, Field

# Local imports
from topollm.config_classes.ConfigBaseModel import ConfigBaseModel

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Globals

# END Globals
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class EmbeddingsConfig(ConfigBaseModel):
    """Configurations for specifying embeddings."""

    huggingface_model_name: str = Field(
        ...,
        title="Model identifier for huggingface transformers model.",
        description="The model identifier for the huggingface transformers model "
        "to use for computing embeddings.",
    )

    layer: str = Field(
        ...,
        title="Layer to use for computing embeddings.",
        description="The layer to use for computing embeddings.",
    )


class DataConfig(ConfigBaseModel):
    """
    Configurations for specifying data.
    """

    column_name: str = Field(
        ...,
        title="Column name to use for computing embeddings.",
        description="The column name to use for computing embeddings.",
    )

    context: str = Field(
        ...,
        title="Context to use for computing embeddings.",
        description="The context to use for computing embeddings.",
    )

    dataset_identifier: str = Field(
        ...,
        title="Dataset identifier for huggingface datasets.",
        description="The dataset identifier for the huggingface datasets "
        "to use for computing embeddings.",
    )

    number_of_samples: int = Field(
        ...,
        title="Number of samples to use for computing embeddings.",
        description="The number of samples to use for computing embeddings.",
    )

    split: str = Field(
        ...,
        title="Split to use for computing embeddings.",
        description="The split to use for computing embeddings.",
    )
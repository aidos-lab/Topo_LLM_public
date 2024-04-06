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

from pydantic import Field

from topollm.config_classes.ConfigBaseModel import ConfigBaseModel
from topollm.config_classes.data.DataConfig import DataConfig
from topollm.config_classes.embeddings.EmbeddingsConfig import EmbeddingsConfig
from topollm.config_classes.enums import PreferredTorchBackend
from topollm.config_classes.finetuning.FinetuningConfig import FinetuningConfig
from topollm.config_classes.inference.InferenceConfig import InferenceConfig
from topollm.config_classes.language_model.LanguageModelConfig import (
    LanguageModelConfig,
)
from topollm.config_classes.PathsConfig import PathsConfig
from topollm.config_classes.StorageConfig import StorageConfig
from topollm.config_classes.tokenizer.TokenizerConfig import TokenizerConfig
from topollm.config_classes.TransformationsConfig import TransformationsConfig

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Globals

# END Globals
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class MainConfig(ConfigBaseModel):
    """
    Master configuration for computing embeddings.
    """

    data: DataConfig = Field(
        ...,
        title="Data configuration.",
        description="The configuration for specifying data.",
    )

    embeddings: EmbeddingsConfig = Field(
        ...,
        title="Embeddings configuration.",
        description="The configuration for specifying embeddings.",
    )

    finetuning: FinetuningConfig = Field(
        ...,
        title="Finetuning configuration.",
        description="The configuration for specifying finetuning.",
    )

    inference: InferenceConfig = Field(
        ...,
        title="Inference configuration.",
        description="The configuration for specifying inference.",
    )

    language_model: LanguageModelConfig = Field(
        ...,
        title="Model configuration.",
        description="The configuration for specifying model.",
    )

    paths: PathsConfig = Field(
        ...,
        title="Paths configuration.",
        description="The configuration for specifying paths.",
    )

    preferred_torch_backend: PreferredTorchBackend = Field(
        ...,
        title="Preferred torch backend.",
        description="The preferred torch backend.",
    )

    seed: int = Field(
        default=1234,
        title="Seed.",
        description="The random seed.",
    )

    storage: StorageConfig = Field(
        ...,
        title="Storage configuration.",
        description="The configuration for specifying storage.",
    )

    tokenizer: TokenizerConfig = Field(
        ...,
        title="Tokenizer configuration.",
        description="The configuration for specifying tokenizer.",
    )

    transformations: TransformationsConfig = Field(
        ...,
        title="Transformations configuration.",
        description="The configuration for specifying transformations.",
    )

    verbosity: int = Field(
        default=1,
        title="Verbosity level.",
        description="The verbosity level.",
    )

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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Imports

# Standard library imports

# Third party imports
from pydantic import Field

# Local imports
from topollm.config_classes.ConfigBaseModel import ConfigBaseModel
from topollm.config_classes.DataConfig import DataConfig

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Globals

# END Globals
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #



class BatchSizesConfig(ConfigBaseModel):
    train: int = Field(
        ...,
        description="The batch size for training.",
    )

    eval: int = Field(
        ...,
        description="The batch size for evaluation.",
    )

class FinetuningDatasetsConfig(ConfigBaseModel):
    train_dataset: DataConfig = Field(
        ...,
        description="The configuration for the training dataset.",
    )

    eval_dataset: DataConfig = Field(
        ...,
        description="The configuration for the evaluation dataset.",
    )

class FinetuningConfig(ConfigBaseModel):
    """Configurations for fine tuning."""

    batch_sizes: BatchSizesConfig = Field(
        ...,
        description="The batch sizes for training and evaluation.",
    )

    finetuning_datasets: FinetuningDatasetsConfig = Field(
        ...,
        description="The configurations for the training and evaluation datasets.",
    )

    fp16: bool = Field(
        ...,
        description="Whether to use 16-bit precision.",
    )

    max_length: int = Field(
        ...,
        description="The maximum length of the input sequence.",
    )

    mlm_probability: float = Field(
        ...,
        description="The probability for masked language model.",
    )

    num_train_epochs: int = Field(
        ...,
        description="The number of training epochs.",
    )

    pretrained_model_name_or_path: str = Field(
        ...,
        description="The name or path of the base model to use for fine tuning.",
    )

    short_model_name: str = Field(
        ...,
        description="Short name of the base model for file names.",
    )




    # TODO
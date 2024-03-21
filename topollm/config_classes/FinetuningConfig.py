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
from regex import F
from torch import log_

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

    eval_steps: int = Field(
        400,
        description="The number of steps between two evaluations.",
    )

    finetuning_datasets: FinetuningDatasetsConfig = Field(
        ...,
        description="The configurations for the training and evaluation datasets.",
    )

    fp16: bool = Field(
        ...,
        description="Whether to use 16-bit precision.",
    )

    gradient_accumulation_steps: int = Field(
        ...,
        description="The number of gradient accumulation steps.",
    )

    gradient_checkpointing: bool = Field(
        ...,
        description="Whether to use gradient checkpointing.",
    )

    learning_rate: float = Field(
        ...,
        description="The learning rate.",
    )

    log_level: str = Field(
        "info",
        description="The log level.",
    )

    logging_steps: int = Field(
        100,
        description="The number of steps between two logging.",
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

    save_steps: int = Field(
        400,
        description="The number of steps between two saves.",
    )

    seed: int = Field(
        ...,
        description="The seed for the random number generator.",
    )

    short_model_name: str = Field(
        ...,
        description="Short name of the base model for file names.",
    )

    use_cpu: bool = Field(
        False,
        description="Whether to use the CPU.",
    )

    warmup_steps: int = Field(
        ...,
        description="The number of warmup steps.",
    )

    weight_decay: float = Field(
        ...,
        description="The weight decay.",
    )

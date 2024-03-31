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
from topollm.config_classes.finetuning.BatchSizesConfig import BatchSizesConfig
from topollm.config_classes.finetuning.FinetuningDatasetsConfig import (
    FinetuningDatasetsConfig,
)
from topollm.config_classes.finetuning.peft.PEFTConfig import PEFTConfig


class FinetuningConfig(ConfigBaseModel):
    """Configurations for fine tuning."""

    peft: PEFTConfig = Field(
        ...,
        description="The configurations for the PEFT model.",
    )

    batch_sizes: BatchSizesConfig = Field(
        ...,
        description="The batch sizes for training and evaluation.",
    )

    finetuning_datasets: FinetuningDatasetsConfig = Field(
        ...,
        description="The configurations for the training and evaluation datasets.",
    )

    eval_steps: int = Field(
        default=400,
        description="The number of steps between two evaluations.",
    )

    fp16: bool = Field(
        default=False,
        description="Whether to use 16-bit precision.",
    )

    gradient_accumulation_steps: int = Field(
        default=2,
        description="The number of gradient accumulation steps.",
    )

    gradient_checkpointing: bool = Field(
        default=True,
        description="Whether to use gradient checkpointing.",
    )

    learning_rate: float = Field(
        default=5e-5,
        description="The learning rate.",
    )

    log_level: str = Field(
        default="info",
        description="The log level.",
    )

    logging_steps: int = Field(
        default=100,
        description="The number of steps between two logging.",
    )

    max_length: int = Field(
        default=512,
        description="The maximum length of the input sequence.",
    )

    max_steps: int = Field(
        default=-1,
        description=f"The maximum number of steps. " f"Overrides num_train_epochs.",
    )

    mlm_probability: float = Field(
        default=0.15,
        description="The probability for masked language model.",
    )

    num_train_epochs: int = Field(
        default=5,
        description="The number of training epochs.",
    )

    pretrained_model_name_or_path: str = Field(
        ...,
        description="The name or path of the base model to use for fine tuning.",
    )

    save_steps: int = Field(
        default=400,
        description="The number of steps between two saves.",
    )

    seed: int = Field(
        default=42,
        description="The seed for the random number generator.",
    )

    short_model_name: str = Field(
        default="default-short-model-name",
        description="Short name of the base model for file names.",
    )

    use_cpu: bool = Field(
        default=False,
        description="Whether to use the CPU.",
    )

    warmup_steps: int = Field(
        default=500,
        description="The number of warmup steps.",
    )

    weight_decay: float = Field(
        default=0.01,
        description="The weight decay.",
    )

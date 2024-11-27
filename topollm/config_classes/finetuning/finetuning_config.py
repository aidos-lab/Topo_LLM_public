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

"""Configuration class for fine tuning."""

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel
from topollm.config_classes.constants import ITEM_SEP, KV_SEP, NAME_PREFIXES
from topollm.config_classes.finetuning.batch_sizes.batch_sizes_config import BatchSizesConfig
from topollm.config_classes.finetuning.finetuning_datasets.finetuning_datasets_config import (
    FinetuningDatasetsConfig,
)
from topollm.config_classes.finetuning.gradient_modifier.gradient_modifier_config import GradientModifierConfig
from topollm.config_classes.finetuning.peft.peft_config import PEFTConfig
from topollm.config_classes.finetuning.trainer_modifier.trainer_modifier_config import TrainerModifierConfig
from topollm.config_classes.language_model.language_model_config import LanguageModelConfig
from topollm.config_classes.tokenizer.tokenizer_config import TokenizerConfig
from topollm.typing.enums import ComputeMetricsMode, DescriptionType, LrSchedulerType


class FinetuningConfig(ConfigBaseModel):
    """Configurations for fine tuning."""

    base_model: LanguageModelConfig = Field(
        default_factory=LanguageModelConfig,
        description="The configuration for specifying the base model.",
    )

    compute_metrics_mode: ComputeMetricsMode = Field(
        default=ComputeMetricsMode.FROM_TASK_TYPE,
        description="The mode for computing metrics.",
    )

    gradient_modifier: GradientModifierConfig = Field(
        default_factory=GradientModifierConfig,
        description="The configurations for the gradient modifier.",
    )

    peft: PEFTConfig = Field(
        default_factory=PEFTConfig,
        description="The configurations for the PEFT model.",
    )

    batch_sizes: BatchSizesConfig = Field(
        default_factory=BatchSizesConfig,
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

    lr_scheduler_type: LrSchedulerType = Field(
        default=LrSchedulerType.LINEAR,
        description="The learning rate scheduler type.",
    )

    log_level: str = Field(
        default="debug",
        description="The log level.",
    )

    logging_steps: int = Field(
        default=100,
        description="The number of steps between two logging.",
    )

    max_steps: int = Field(
        default=-1,
        description="The maximum number of steps. Overrides num_train_epochs.",
    )

    mlm_probability: float = Field(
        default=0.15,
        description="The probability for masked language model.",
    )

    num_train_epochs: int = Field(
        default=5,
        description="The number of training epochs.",
    )

    save_steps: int = Field(
        default=400,
        description="The number of steps between two saves.",
    )

    seed: int = Field(
        default=1234,
        description="The seed used in finetuning for the Trainer.",
    )

    tokenizer: TokenizerConfig = Field(
        default_factory=TokenizerConfig,
        title="Tokenizer configuration.",
        description="The configuration for specifying tokenizer.",
    )

    trainer_modifier: TrainerModifierConfig = Field(
        default_factory=TrainerModifierConfig,
        description="The configurations for the trainer modifier.",
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

    report_to: list[str] = Field(
        default=[
            "wandb",
            "tensorboard",
        ],
        description="The reporting tool.",
    )

    def get_base_model_config_description(
        self,
        description_type: DescriptionType = DescriptionType.LONG,
        short_description_separator: str = "-",
    ) -> str:
        """Construct and return the model description."""
        description: str = self.base_model.get_config_description(
            description_type=description_type,
            short_description_separator=short_description_separator,
        )

        return description

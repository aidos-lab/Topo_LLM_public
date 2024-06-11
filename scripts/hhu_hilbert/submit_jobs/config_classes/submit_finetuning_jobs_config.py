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

import os
import pathlib
from dataclasses import dataclass, field

TOPO_LLM_REPOSITORY_BASE_PATH = os.getenv(
    "TOPO_LLM_REPOSITORY_BASE_PATH",
    "$HOME/git-source/Topo_LLM",
)


@dataclass
class TrainingScheduleConfig:
    """Config for training schedule."""

    num_train_epochs: int = 5
    lr_scheduler_type: str = "linear"


@dataclass
class LoraParameters:
    lora_r: int
    lora_alpha: int
    use_rslora: bool = False


@dataclass
class SubmitFinetuningJobsConfig:
    """Config for submitting finetuning jobs."""

    base_model: list[str]
    finetuning_dataset: list[str]
    peft: list[str]
    gradient_modifier: list[str]
    lora_parameters: dict[str, LoraParameters]
    training_schedule: dict[str, TrainingScheduleConfig]
    common_batch_size: int = 16
    save_steps: int = 400
    eval_steps: int = 100
    fp16: str = "true"

    finetuning_python_script_relative_path: pathlib.Path = pathlib.Path(
        "topollm",
        "model_finetuning",
        "run_finetune_language_model_on_huggingface_dataset.py",
    )

    topo_llm_repository_base_path: str = TOPO_LLM_REPOSITORY_BASE_PATH

    wandb_project: str = "Topo_LLM_submit_jobs_via_hydra_debug"

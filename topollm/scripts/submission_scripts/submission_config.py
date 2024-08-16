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

"""Configuration for submitting a certain job."""

import pathlib

from pydantic import BaseModel, Field

from topollm.config_classes.constants import TOPO_LLM_REPOSITORY_BASE_PATH
from topollm.typing.enums import SubmissionMode, Task


class SubmissionConfig(BaseModel):
    """Configuration for submitting a certain job."""

    # Submission parameters
    queue: str | None = "DSML"
    template: str | None = "DSML"
    memory: str | None = "64"
    submission_mode: SubmissionMode = SubmissionMode.HPC_SUBMISSION

    # Common parameters
    add_prefix_space: bool = False
    data_list: list[str] = Field(
        default_factory=lambda: [
            "iclr_2024_submissions_validation",
            "multiwoz21_validation",
        ],
    )
    language_model_list: list[str] = Field(
        default_factory=lambda: [
            "roberta-base",
        ],
    )
    checkpoint_no: str | None = None

    layer_indices: str | None = "[-1]"
    embeddings_data_prep_num_samples: str | None = "30000"
    embeddings_data_prep_sampling_mode: str | None = "take_first"
    additional_overrides: str | None = ""

    # Finetuning-specific parameters
    base_model_list: list[str] = Field(
        default_factory=lambda: [
            "roberta-base_for_masked_lm",
        ],
    )
    num_train_epochs: str | None = "5"
    save_steps: str | None = "400"
    eval_steps: str | None = "100"
    finetuning_datasets_list: list[str] = Field(
        default_factory=lambda: [
            "multiwoz21_train",
        ],
    )
    lr_scheduler_type: str | None = "linear"
    peft_list: str | None = "standard"
    gradient_modifier_list: str | None = "do_nothing"
    batch_size_train: str | None = "8"
    batch_size_eval: str | None = "8"

    python_script_name: str = Field(
        default="run_pipeline_compute_embeddings_and_data_prep_and_local_estimate.py",
    )

    relative_python_script_folder: pathlib.Path = Field(
        default=pathlib.Path(
            "topollm",
            "pipeline_scripts",
        ),
    )

    @property
    def relative_python_script_path(
        self,
    ) -> pathlib.Path:
        return pathlib.Path(
            self.relative_python_script_folder,
            self.python_script_name,
        )

    @property
    def absolute_python_script_path(
        self,
    ) -> pathlib.Path:
        return pathlib.Path(
            TOPO_LLM_REPOSITORY_BASE_PATH,
            self.relative_python_script_path,
        )

    @property
    def poetry_run_command(
        self,
    ) -> list[str]:
        return [
            "poetry",
            "run",
            "python3",
            str(self.relative_python_script_path),
            "--multirun",
            "hydra/sweeper=basic",
        ]

    def get_command(
        self,
        task: Task,
    ) -> list[str]:
        """Get the command to run the task."""
        match task:
            case Task.PIPELINE:
                task_specific_command = [
                    f"data={','.join(self.data_list)}",
                    f"language_model={','.join(self.language_model_list)}",
                    f"tokenizer.add_prefix_space={self.add_prefix_space}",
                ]

                if self.checkpoint_no:
                    task_specific_command.append(
                        f"language_model.checkpoint_no={self.checkpoint_no}",
                    )

                if self.layer_indices:
                    task_specific_command.append(
                        f"embeddings.embedding_extraction.layer_indices={self.layer_indices}",
                    )

                if self.embeddings_data_prep_num_samples:
                    task_specific_command.append(
                        f"embeddings_data_prep.sampling.num_samples={self.embeddings_data_prep_num_samples}",
                    )

                if self.embeddings_data_prep_sampling_mode:
                    task_specific_command.append(
                        f"+embeddings_data_prep.sampling.sampling_mode={self.embeddings_data_prep_sampling_mode}",
                    )
            case Task.PERPLEXITY:
                task_specific_command = [
                    f"data={','.join(self.data_list)}",
                    f"language_model={','.join(self.language_model_list)}",
                    f"tokenizer.add_prefix_space={self.add_prefix_space}",
                ]
                if self.checkpoint_no:
                    task_specific_command.append(
                        f"language_model.checkpoint_no={self.checkpoint_no}",
                    )

            case Task.FINETUNING:
                task_specific_command = [
                    f"finetuning/base_model={','.join(self.base_model_list)}",
                    f"finetuning.num_train_epochs={self.num_train_epochs}",
                    f"finetuning.lr_scheduler_type={self.lr_scheduler_type}",
                    f"finetuning.save_steps={self.save_steps}",
                    f"finetuning.eval_steps={self.eval_steps}",
                    "finetuning.fp16=true",
                    f"finetuning.batch_sizes.train={self.batch_size_train}",
                    f"finetuning.batch_sizes.eval={self.batch_size_eval}",
                    f"finetuning/finetuning_datasets={','.join(self.finetuning_datasets_list)}",
                    f"finetuning/peft={self.peft_list}",
                    f"finetuning/gradient_modifier={self.gradient_modifier_list}",
                ]
            case _:
                msg = f"Unknown {task = }"
                raise ValueError(msg)

        # Assemble the command
        command: list[str] = self.poetry_run_command + task_specific_command

        if self.submission_mode == SubmissionMode.HPC_SUBMISSION:
            command.extend(
                [
                    f"hydra/launcher={self.submission_mode.value}",
                    f"hydra.launcher.queue={self.queue}",
                    f"hydra.launcher.template={self.template}",
                    f"hydra.launcher.memory={self.memory}",
                ],
            )

        if self.additional_overrides:
            command.append(self.additional_overrides)

        return command

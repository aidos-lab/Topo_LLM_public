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

import argparse
import pathlib
import subprocess
from enum import StrEnum, auto

from pydantic import BaseModel, Field

from topollm.config_classes.constants import TOPO_LLM_REPOSITORY_BASE_PATH


class SubmissionMode(StrEnum):
    """Submission mode for running scripts."""

    LOCAL = auto()
    HPC_SUBMISSION = auto()


class Task(StrEnum):
    """Enumeration of tasks."""

    PIPELINE = auto()
    PERPLEXITY = auto()
    FINETUNING = auto()


class SubmissionConfig(BaseModel):
    """Configuration for submitting a certain job."""

    # Submission parameters
    queue: str | None = "DSML"
    template: str | None = "DSML"
    submission_mode: SubmissionMode = SubmissionMode.HPC_SUBMISSION

    # Common parameters
    add_prefix_space: bool = False
    data_lists: list[str] = Field(
        default_factory=lambda: [
            "multiwoz21_validation",
        ],
    )
    language_models: list[str] = Field(
        default_factory=lambda: [
            "roberta-base",
        ],
    )
    layer_indices_list: str | None = "[-1]"
    embeddings_data_prep_num_samples: str | None = "30000"
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

    def get_command(
        self,
        task: Task,
    ) -> list[str]:
        match task:
            case Task.PIPELINE:
                command = [
                    "poetry",
                    "run",
                    "python3",
                    str(self.relative_python_script_path),
                    "--multirun",
                    "hydra/sweeper=basic",
                    f"data={','.join(self.data_lists)}",
                    f"language_model={','.join(self.language_models)}",
                    f"tokenizer.add_prefix_space={self.add_prefix_space}",
                ]

                if self.layer_indices_list:
                    command.append(f"embeddings.embedding_extraction.layer_indices={self.layer_indices_list}")

                if self.embeddings_data_prep_num_samples:
                    command.append(f"embeddings_data_prep.sampling.num_samples={self.embeddings_data_prep_num_samples}")
            case Task.PERPLEXITY:
                command = [
                    "poetry",
                    "run",
                    "python3",
                    str(self.relative_python_script_path),
                    "--multirun",
                    "hydra/sweeper=basic",
                    f"data={','.join(self.data_lists)}",
                    f"language_model={','.join(self.language_models)}",
                    f"tokenizer.add_prefix_space={self.add_prefix_space}",
                ]
            case Task.FINETUNING:
                command = [
                    "poetry",
                    "run",
                    "python3",
                    str(self.relative_python_script_path),
                    "--multirun",
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

        if self.submission_mode == SubmissionMode.HPC_SUBMISSION:
            command.extend(
                [
                    f"hydra/launcher={self.submission_mode.value}",
                    f"hydra.launcher.queue={self.queue}",
                    f"hydra.launcher.template={self.template}",
                ],
            )

        if self.additional_overrides:
            command.append(self.additional_overrides)

        return command


def run_task(
    submissions_config: SubmissionConfig,
    task: Task,
) -> None:
    """Run a task with the given configuration."""
    match task:
        case Task.PIPELINE:
            submissions_config.python_script_name = (
                "run_pipeline_compute_embeddings_and_data_prep_and_local_estimate.py"
            )
            submissions_config.relative_python_script_folder = pathlib.Path(
                "topollm",
                "pipeline_scripts",
            )
        case Task.PERPLEXITY:
            submissions_config.python_script_name = "run_compute_perplexity.py"
            submissions_config.relative_python_script_folder = pathlib.Path(
                "topollm",
                "model_inference",
                "perplexity",
            )
        case Task.FINETUNING:
            submissions_config.python_script_name = "run_finetune_language_model_on_huggingface_dataset.py"
            submissions_config.relative_python_script_folder = pathlib.Path(
                "topollm",
                "model_finetuning",
            )
        case _:
            msg = f"Unknown {task = }"
            raise ValueError(msg)

    command = submissions_config.get_command(
        task=task,
    )
    print(  # noqa: T201 - We want this submission script to print the command
        "Running command:",
        " ".join(command),
    )
    subprocess.run(  # noqa: S603 - We trust the command
        command,
        check=True,
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run computations for pipeline, perplexity, or finetuning.",
    )

    parser.add_argument(
        "--task",
        type=Task,
        default=Task.PIPELINE,
        help="Specify the task to run.",
    )
    parser.add_argument(
        "--submission_mode",
        type=SubmissionMode,
        default=SubmissionMode.HPC_SUBMISSION,
        help="Submission mode.",
    )
    parser.add_argument(
        "--queue",
        type=str,
        default="DSML",
        help="Queue name for HPC submission.",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="DSML",
        help="Template name for HPC submission.",
    )
    parser.add_argument(
        "--additional-overrides",
        type=str,
        default="",
        help="Additional overrides for Hydra.",
    )

    args = parser.parse_args()
    return args


def make_config_and_run_task(
    args: argparse.Namespace,
) -> None:
    """Make a submission configuration and run the task."""
    submissions_config = SubmissionConfig(
        submission_mode=args.submission_mode,
        queue=args.queue,
        template=args.template,
        additional_overrides=args.additional_overrides,
    )

    run_task(
        submissions_config=submissions_config,
        task=args.task,
    )


def main() -> None:
    """Run the submission."""
    args = parse_arguments()

    make_config_and_run_task(
        args=args,
    )


if __name__ == "__main__":
    main()

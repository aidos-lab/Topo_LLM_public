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

"""Submit jobs for pipeline, perplexity, or finetuning."""

import argparse
import pathlib
import subprocess
from enum import StrEnum, auto

from topollm.scripts.submission_scripts.submission_config import SubmissionConfig
from topollm.typing.enums import SubmissionMode, Task


def run_task(
    submissions_config: SubmissionConfig,
    task: Task,
    *,
    dry_run: bool = False,
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
        "Running command:\n",
        " ".join(command),
    )

    if dry_run:
        print(  # noqa: T201 - We want this submission script to print this output
            "@@@@ Dry run, not actually running the command. @@@@",
        )
    else:
        subprocess.run(  # noqa: S603 - We trust the command
            command,
            check=True,
        )


class DataListOption(StrEnum):
    """Options for the data list."""

    DEBUG = auto()
    FULL = auto()
    MANUAL_IN_PYTHON_SCRIPT = auto()
    TRAIN_ONLY = auto()
    MULTIWOZ21_AND_REDDIT = auto()
    MULTIWOZ21_ONLY = auto()


class FinetuningDatasetsListOption(StrEnum):
    """Options for the finetuning dataset list."""

    DEBUG = auto()
    MANUAL_IN_PYTHON_SCRIPT = auto()


class LanguageModelListOption(StrEnum):
    """Options for the language model list."""

    ONLY_ROBERTA_BASE = auto()
    SELECTED_FINETUNED_FEW_EPOCHS_FROM_ROBERTA_BASE = auto()
    SELECTED_FINETUNED_MANY_EPOCHS_FROM_ROBERTA_BASE = auto()
    FULL_FINETUNED_FEW_EPOCHS_FROM_ROBERTA_BASE = auto()
    SETSUMBT_SELECTED = auto()


class CheckpointNoListOption(StrEnum):
    """Options for the checkpoint number list."""

    SELECTED = auto()
    FULL = auto()


class FinetuningRegimeOption(StrEnum):
    """Options for the finetuning regime."""

    FEW_EPOCHS = auto()
    MANY_EPOCHS_WITH_OVERFITTING_RISK = auto()


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
        "--additional_overrides",
        type=str,
        default="",
        help="Additional overrides for Hydra.",
    )

    # Selecting groups of data and language models
    parser.add_argument(
        "--data_list",
        type=DataListOption,
        default=DataListOption.DEBUG,
        help="Data list to use.",
    )
    parser.add_argument(
        "--language_model_list",
        type=LanguageModelListOption,
        default=LanguageModelListOption.SELECTED_FINETUNED_FEW_EPOCHS_FROM_ROBERTA_BASE,
        help="Language model list to use.",
    )
    parser.add_argument(
        "--finetuning_datasets_list",
        type=FinetuningDatasetsListOption,
        default=FinetuningDatasetsListOption.DEBUG,
        help="Finetuning datasets list to use.",
    )
    parser.add_argument(
        "--checkpoint_no_list",
        type=CheckpointNoListOption,
        default=CheckpointNoListOption.SELECTED,
        help="Checkpoint number list to use.",
    )
    parser.add_argument(
        "--finetuning_regime",
        type=FinetuningRegimeOption,
        default=FinetuningRegimeOption.FEW_EPOCHS,
        help="Finetuning regime to use.",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run the command.",
    )

    args = parser.parse_args()
    return args


def make_config_and_run_task(
    args: argparse.Namespace,
) -> None:
    """Make a submission configuration and run the task."""
    full_data_list = [
        "iclr_2024_submissions_test",
        "iclr_2024_submissions_train",
        "iclr_2024_submissions_validation",
        "multiwoz21_test",
        "multiwoz21_train",
        "multiwoz21_validation",
        "one-year-of-tsla-on-reddit_test",
        "one-year-of-tsla-on-reddit_train",
        "one-year-of-tsla-on-reddit_validation",
        "sgd_test",
        "sgd_train",
        "sgd_validation",
        "wikitext_test",
        "wikitext_train",
        "wikitext_validation",
    ]
    only_train_data_list = [data_name for data_name in full_data_list if "_train" in data_name]

    only_roberta_base_language_model_list = [
        "roberta-base",
    ]
    selected_finetuned_few_epochs_from_roberta_base_language_model_list = [
        "model-roberta-base_task-masked_lm_multiwoz21-train-10000-ner_tags_ftm-standard_lora-None_5e-05-linear-0.01-5",
        "model-roberta-base_task-masked_lm_one-year-of-tsla-on-reddit-train-10000-ner_tags_ftm-standard_lora-None_5e-05-linear-0.01-5",
    ]
    selected_finetuned_many_epochs_from_roberta_base_language_model_list = [
        "model-roberta-base_task-masked_lm_multiwoz21-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50",
        "model-roberta-base_task-masked_lm_one-year-of-tsla-on-reddit-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50",
    ]
    full_finetuned_few_epochs_from_roberta_base_language_model_list = [
        "model-roberta-base_task-masked_lm_iclr_2024_submissions-train-5000-ner_tags_ftm-standard_lora-None_5e-05-linear-0.01-5",
        "model-roberta-base_task-masked_lm_multiwoz21-train-10000-ner_tags_ftm-standard_lora-None_5e-05-linear-0.01-5",
        "model-roberta-base_task-masked_lm_one-year-of-tsla-on-reddit-train-10000-ner_tags_ftm-standard_lora-None_5e-05-linear-0.01-5",
        "model-roberta-base_task-masked_lm_wikitext-train-10000-ner_tags_ftm-standard_lora-None_5e-05-linear-0.01-5",
    ]

    setsumbt_model_list = [
        "model-roberta-base_task-setsumbt_multiwoz21",
    ]

    match args.data_list:
        case DataListOption.FULL:
            data_list = full_data_list
        case DataListOption.TRAIN_ONLY:
            data_list = only_train_data_list
        case DataListOption.DEBUG:
            data_list = [
                "multiwoz21_test",
                "sgd_test",
            ]
        case DataListOption.MANUAL_IN_PYTHON_SCRIPT:
            data_list = [
                "iclr_2024_submissions_test",
                "multiwoz21_test",
                "one-year-of-tsla-on-reddit_test",
                "sgd_test",
                "wikitext_test",
            ]
        case DataListOption.MULTIWOZ21_AND_REDDIT:
            data_list = [
                "multiwoz21_test",
                "multiwoz21_train",
                "multiwoz21_validation",
                "one-year-of-tsla-on-reddit_test",
                "one-year-of-tsla-on-reddit_train",
                "one-year-of-tsla-on-reddit_validation",
            ]
        case DataListOption.MULTIWOZ21_ONLY:
            data_list = [
                "multiwoz21_test",
                "multiwoz21_train",
                "multiwoz21_validation",
            ]
        case _:
            msg = f"Unknown {args.data_list = }"
            raise ValueError(msg)

    match args.finetuning_regime:
        case FinetuningRegimeOption.FEW_EPOCHS:
            num_train_epochs = "5"
            lr_scheduler_type = "linear"
        case FinetuningRegimeOption.MANY_EPOCHS_WITH_OVERFITTING_RISK:
            num_train_epochs = "50"
            lr_scheduler_type = "constant"
        case _:
            msg = f"Unknown {args.finetuning_regime = }"
            raise ValueError(msg)

    match args.language_model_list:
        case LanguageModelListOption.ONLY_ROBERTA_BASE:
            language_model_list = only_roberta_base_language_model_list
            # No checkpoints for the base model
            checkpoint_no_list = None
        case LanguageModelListOption.SELECTED_FINETUNED_FEW_EPOCHS_FROM_ROBERTA_BASE:
            language_model_list = selected_finetuned_few_epochs_from_roberta_base_language_model_list

            checkpoint_no_list = get_checkpoint_no_list(
                checkpoint_no_list_option=args.checkpoint_no_list,
                num_train_epochs=int(num_train_epochs),
            )
        case LanguageModelListOption.FULL_FINETUNED_FEW_EPOCHS_FROM_ROBERTA_BASE:
            language_model_list = full_finetuned_few_epochs_from_roberta_base_language_model_list

            checkpoint_no_list = get_checkpoint_no_list(
                checkpoint_no_list_option=args.checkpoint_no_list,
                num_train_epochs=int(num_train_epochs),
            )
        case LanguageModelListOption.SELECTED_FINETUNED_MANY_EPOCHS_FROM_ROBERTA_BASE:
            language_model_list = selected_finetuned_many_epochs_from_roberta_base_language_model_list

            checkpoint_no_list = get_checkpoint_no_list(
                checkpoint_no_list_option=args.checkpoint_no_list,
                num_train_epochs=int(num_train_epochs),
            )
        case LanguageModelListOption.SETSUMBT_SELECTED:
            language_model_list = setsumbt_model_list

            # No checkpoints for the setsumbt model
            checkpoint_no_list = [
                "2813",
                "5626",
                "8439",
            ]
        case _:
            msg = f"Unknown {args.language_model_list = }"
            raise ValueError(msg)

    match args.finetuning_datasets_list:
        case FinetuningDatasetsListOption.DEBUG:
            finetuning_datasets_list = [
                "train_and_eval_on_multiwoz21_train-samples-small",
            ]
        case FinetuningDatasetsListOption.MANUAL_IN_PYTHON_SCRIPT:
            finetuning_datasets_list = [
                "train_and_eval_on_multiwoz21_train-samples-small",
                "train_and_eval_on_one-year-of-tsla-on-reddit_train-samples-small",
            ]
        case _:
            msg = f"Unknown {args.finetuning_datasets_list = }"
            raise ValueError(msg)

    submissions_config = SubmissionConfig(
        submission_mode=args.submission_mode,
        queue=args.queue,
        template=args.template,
        additional_overrides=args.additional_overrides,
        data_list=data_list,
        language_model_list=language_model_list,
        checkpoint_no_list=checkpoint_no_list,
        finetuning_datasets_list=finetuning_datasets_list,
        num_train_epochs=num_train_epochs,
        lr_scheduler_type=lr_scheduler_type,
    )

    run_task(
        submissions_config=submissions_config,
        task=args.task,
        dry_run=args.dry_run,
    )


def get_checkpoint_no_list(
    checkpoint_no_list_option: CheckpointNoListOption,
    num_train_epochs: int = 5,
) -> list[str]:
    """Get the list of checkpoint numbers to use."""
    # TODO: Make this more flexible (work for other numbers of epochs)
    match checkpoint_no_list_option:
        case CheckpointNoListOption.SELECTED:
            if num_train_epochs == 5:
                checkpoint_no_list = [
                    "400",
                    "1200",
                    "2000",
                    "2800",
                ]
            elif num_train_epochs == 50:
                checkpoint_no_list = [
                    "400",
                    "3200",
                    "6000",
                    "8800",
                    "11600",
                    "14400",
                    "17200",
                    "20000",
                    "22800",
                    "25600",
                    "28400",
                    "31200",
                ]
            else:
                msg = f"Unknown {num_train_epochs = }"
                raise ValueError(msg)
        case CheckpointNoListOption.FULL:
            if num_train_epochs == 5:
                # All checkpoints from 400 to 2800
                # (for ep-5 and batch size 8)
                checkpoint_no_list = [
                    "400",
                    "800",
                    "1200",
                    "1600",
                    "2000",
                    "2400",
                    "2800",
                ]
            elif num_train_epochs == 50:
                # All checkpoints from 400 to 31200
                # (for ep-50 and batch size 8)
                checkpoint_no_list = [
                    "400",
                    "800",
                    "1200",
                    "1600",
                    "2000",
                    "2400",
                    "2800",
                    "3200",
                    "3600",
                    "4000",
                    "4400",
                    "4800",
                    "5200",
                    "5600",
                    "6000",
                    "6400",
                    "6800",
                    "7200",
                    "7600",
                    "8000",
                    "8400",
                    "8800",
                    "9200",
                    "9600",
                    "10000",
                    "10400",
                    "10800",
                    "11200",
                    "11600",
                    "12000",
                    "12400",
                    "12800",
                    "13200",
                    "13600",
                    "14000",
                    "14400",
                    "14800",
                    "15200",
                    "15600",
                    "16000",
                    "16400",
                    "16800",
                    "17200",
                    "17600",
                    "18000",
                    "18400",
                    "18800",
                    "19200",
                    "19600",
                    "20000",
                    "20400",
                    "20800",
                    "21200",
                    "21600",
                    "22000",
                    "22400",
                    "22800",
                    "23200",
                    "23600",
                    "24000",
                    "24400",
                    "24800",
                    "25200",
                    "25600",
                    "26000",
                    "26400",
                    "26800",
                    "27200",
                    "27600",
                    "28000",
                    "28400",
                    "28800",
                    "29200",
                    "29600",
                    "30000",
                    "30400",
                    "30800",
                    "31200",
                ]
            else:
                msg = f"Unknown {num_train_epochs = }"
                raise ValueError(msg)
        case _:
            msg = f"Unknown {checkpoint_no_list_option = }"
            raise ValueError(msg)

    return checkpoint_no_list


def main() -> None:
    """Run the submission."""
    args = parse_arguments()

    make_config_and_run_task(
        args=args,
    )


if __name__ == "__main__":
    main()

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

from topollm.scripts.submission_scripts.get_checkpoint_no_list import get_checkpoint_no_list
from topollm.scripts.submission_scripts.submission_config import SubmissionConfig
from topollm.scripts.submission_scripts.types import (
    CheckpointNoListOption,
    DataListOption,
    DataNumberOfSamplesListOption,
    EmbeddingsDataPrepNumSamplesListOption,
    EmbeddingsDataPrepSamplingSeedListOption,
    FinetuningDatasetsListOption,
    FinetuningRegimeOption,
    LanguageModelListOption,
    LocalEstimatesFilteringNumSamplesListOption,
    LocalEstimatesPointwiseAbsoluteNNeighborsListOption,
    SeedListOption,
)
from topollm.typing.enums import EmbeddingsDataPrepSamplingMode, SubmissionMode, Task


def run_task(
    submissions_config: SubmissionConfig,
    task: Task,
    *,
    dry_run: bool = False,
) -> None:
    """Run a task with the given configuration."""
    match task:
        case Task.LOCAL_ESTIMATES_COMPUTATION:
            submissions_config.python_script_name = "run_local_estimates.py"
            submissions_config.relative_python_script_folder = pathlib.Path(
                "topollm",
                "analysis",
                "local_estimates_computation",
            )
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

    command: list[str] = submissions_config.get_command(
        task=task,  # type: ignore - StrEnum typing problems
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
        subprocess.run(
            args=command,
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
        default="",
        help="Queue name for HPC submission.",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="Template name for HPC submission.",
    )
    parser.add_argument(
        "--additional_overrides",
        type=str,
        default="",
        help="Additional overrides for Hydra.",
    )
    parser.add_argument(
        "--memory",
        type=str,
        default=None,
        help="Memory to use.",
    )
    parser.add_argument(
        "--ncpus",
        type=str,
        default=None,
        help="Number of CPUs.",
    )
    parser.add_argument(
        "--ngpus",
        type=str,
        default=None,
        help="Number of GPUs.",
    )
    parser.add_argument(
        "--walltime",
        type=str,
        default="8:00:00",
        help="Walltime.",
    )

    # Selecting groups of data and language models
    parser.add_argument(
        "--data_list",
        type=DataListOption,
        default=DataListOption.DEBUG,
        help="Data list to use.",
    )
    parser.add_argument(
        "--data_number_of_samples_list_option",
        type=DataNumberOfSamplesListOption,
        default=DataNumberOfSamplesListOption.NONE,
        help="data_number_of_samples_list option to use.",
    )
    parser.add_argument(
        "--language_model_list",
        type=LanguageModelListOption,
        default=LanguageModelListOption.SELECTED_FINETUNED_FEW_EPOCHS_FROM_ROBERTA_BASE,
        help="Language model list to use.",
    )
    parser.add_argument(
        "--language_model_seed_list",
        type=SeedListOption,
        default=SeedListOption.DO_NOT_SET,
        help="Language model seed list to use.",
    )
    parser.add_argument(
        "--finetuning_datasets_list",
        type=FinetuningDatasetsListOption,
        default=FinetuningDatasetsListOption.DEBUG,
        help="Finetuning datasets list to use.",
    )
    parser.add_argument(
        "--finetuning_seed_list",
        type=SeedListOption,
        default=SeedListOption.DO_NOT_SET,
        help="Finetuning seed list to use.",
    )
    parser.add_argument(
        "--checkpoint_no_list",
        type=CheckpointNoListOption,
        default=CheckpointNoListOption.SELECTED,
        help="Checkpoint number list to use.",
    )
    parser.add_argument(
        "--embeddings_data_prep_num_samples_list_option",
        type=EmbeddingsDataPrepNumSamplesListOption,
        default=EmbeddingsDataPrepNumSamplesListOption.DEFAULT,
        help="Embeddings data prep number of samples list to use.",
    )
    parser.add_argument(
        "--embeddings_data_prep_sampling_mode",
        type=EmbeddingsDataPrepSamplingMode,
        default=EmbeddingsDataPrepSamplingMode.RANDOM,
        help="Embeddings data prep sampling mode to use.",
    )
    parser.add_argument(
        "--embeddings_data_prep_sampling_seed_list_option",
        type=EmbeddingsDataPrepSamplingSeedListOption,
        default=EmbeddingsDataPrepSamplingSeedListOption.DEFAULT,
        help="Embeddings data prep sampling seed list option to use.",
    )

    parser.add_argument(
        "--finetuning_regime",
        type=FinetuningRegimeOption,
        default=FinetuningRegimeOption.FEW_EPOCHS,
        help="Finetuning regime to use.",
    )
    parser.add_argument(
        "--add_prefix_space",
        action="store_true",
        help="Set the add_prefix_space option.",
    )

    parser.add_argument(
        "--create_pos_tags",
        action="store_true",
        help="Whether to create POS tags in the dataset.",
    )

    parser.add_argument(
        "--local_estimates_filtering_num_samples_list",
        type=LocalEstimatesFilteringNumSamplesListOption,
        default=LocalEstimatesFilteringNumSamplesListOption.DEFAULT,
        help="Local estimates filtering number of samples list to use.",
    )
    parser.add_argument(
        "--local_estimates_pointwise_absolute_n_neighbors_list",
        type=LocalEstimatesPointwiseAbsoluteNNeighborsListOption,
        default=LocalEstimatesPointwiseAbsoluteNNeighborsListOption.DEFAULT,
        help="Local estimates pointwise absolute n neighbors list to use.",
    )
    parser.add_argument(
        "--skip_compute_and_store_embeddings",
        action="store_true",
        help="Skip the compute and store embeddings step.",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run the command.",
    )

    args = parser.parse_args()
    return args


full_data_list: list[str] = [
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
only_train_data_list: list[str] = [data_name for data_name in full_data_list if "_train" in data_name]

only_roberta_base_language_model_list: list[str] = [
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

seed_list_option_one_seed: list[str] = [
    "1234",
]

seed_list_option_two_seeds: list[str] = [
    "1234",
    "1235",
]
seed_list_option_five_seeds: list[str] = [
    "1234",
    "1235",
    "1236",
    "1237",
    "1238",
]


def make_config_and_run_task(
    args: argparse.Namespace,
) -> None:
    """Make a submission configuration and run the task."""
    match args.data_list:
        case DataListOption.FULL:
            data_list = full_data_list
        case DataListOption.TRAIN_ONLY:
            data_list = only_train_data_list
        case DataListOption.DEBUG:
            data_list: list[str] = [
                "multiwoz21_test",
                "sgd_test",
            ]
        case DataListOption.MANUAL_IN_PYTHON_SCRIPT:
            data_list: list[str] = [
                "iclr_2024_submissions_test",
                "multiwoz21_test",
                "one-year-of-tsla-on-reddit_test",
                "sgd_test",
                "wikitext_test",
            ]
        case DataListOption.MULTIWOZ21_AND_REDDIT:
            data_list: list[str] = [
                "multiwoz21_test",
                "multiwoz21_train",
                "multiwoz21_validation",
                "one-year-of-tsla-on-reddit_test",
                "one-year-of-tsla-on-reddit_train",
                "one-year-of-tsla-on-reddit_validation",
            ]
        case DataListOption.MULTIWOZ21_TRAIN_AND_REDDIT_TRAIN:
            data_list: list[str] = [
                "multiwoz21_train",
                "one-year-of-tsla-on-reddit_train",
            ]
        case DataListOption.MULTIWOZ21_ONLY:
            data_list: list[str] = [
                "multiwoz21_test",
                "multiwoz21_train",
                "multiwoz21_validation",
            ]
        case DataListOption.REDDIT_ONLY:
            data_list: list[str] = [
                "one-year-of-tsla-on-reddit_test",
                "one-year-of-tsla-on-reddit_train",
                "one-year-of-tsla-on-reddit_validation",
            ]
        case _:
            msg = f"Unknown {args.data_list = }"
            raise ValueError(
                msg,
            )

    match args.data_number_of_samples_list_option:
        case DataNumberOfSamplesListOption.NONE:
            data_number_of_samples_list = None
        case DataNumberOfSamplesListOption.FIXED_3000:
            data_number_of_samples_list = [
                "3000",
            ]
        case DataNumberOfSamplesListOption.FIXED_10000:
            data_number_of_samples_list = [
                "10000",
            ]
        case _:
            msg = f"Unknown {args.data_number_of_samples_list_option = }"
            raise ValueError(
                msg,
            )

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
            setsumbt_seed: int = 0
            language_model_list = setsumbt_model_list

            # Note: For the different seeds in the SETSUMBT training,
            # we have saved checkpoints at different global steps.
            if setsumbt_seed == 0:
                checkpoint_no_list = [
                    "2813",
                    "5626",
                    "8439",
                    "11252",
                    "14065",
                    "16878",
                    "19691",
                    "25317",
                    "33756",
                    "36569",
                    "39382",
                    "42195",
                    "50634",
                    "56260",
                    "70325",
                    "90016",
                    "109707",
                    "115333",
                    "126585",
                ]
            else:
                msg = f"Unknown {setsumbt_seed = }"
                raise ValueError(msg)
        case _:
            msg = f"Unknown {args.language_model_list = }"
            raise ValueError(
                msg,
            )

    match args.language_model_seed_list:
        case SeedListOption.DO_NOT_SET:
            language_model_seed_list = None
        case SeedListOption.ONE_SEED:
            language_model_seed_list = seed_list_option_one_seed
        case SeedListOption.TWO_SEEDS:
            language_model_seed_list = seed_list_option_two_seeds
        case SeedListOption.FIVE_SEEDS:
            language_model_seed_list = seed_list_option_five_seeds
        case _:
            msg: str = f"Unknown {args.language_model_seed_list = }"
            raise ValueError(
                msg,
            )

    match args.finetuning_datasets_list:
        case FinetuningDatasetsListOption.DEBUG:
            finetuning_datasets_list: list[str] = [
                "train_and_eval_on_multiwoz21_train-samples-small",
            ]
        case FinetuningDatasetsListOption.MANUAL_IN_PYTHON_SCRIPT:
            finetuning_datasets_list: list[str] = [
                "train_and_eval_on_multiwoz21_train-samples-small",
                "train_and_eval_on_one-year-of-tsla-on-reddit_train-samples-small",
            ]
        case _:
            msg = f"Unknown {args.finetuning_datasets_list = }"
            raise ValueError(
                msg,
            )

    match args.finetuning_seed_list:
        case SeedListOption.DO_NOT_SET:
            finetuning_seed_list = None
        case SeedListOption.TWO_SEEDS:
            finetuning_seed_list = seed_list_option_two_seeds
        case SeedListOption.FIVE_SEEDS:
            finetuning_seed_list = seed_list_option_five_seeds
        case _:
            msg: str = f"Unknown {args.finetuning_seed_list = }"
            raise ValueError(
                msg,
            )

    match args.embeddings_data_prep_sampling_seed_list_option:
        case EmbeddingsDataPrepSamplingSeedListOption.DEFAULT:
            embeddings_data_prep_sampling_seed_list = [
                "42",
            ]
        case EmbeddingsDataPrepSamplingSeedListOption.TWO_SEEDS:
            embeddings_data_prep_sampling_seed_list = [
                "42",
                "43",
            ]
        case EmbeddingsDataPrepSamplingSeedListOption.FIVE_SEEDS:
            embeddings_data_prep_sampling_seed_list = [str(i) for i in range(42, 47)]
        case EmbeddingsDataPrepSamplingSeedListOption.TEN_SEEDS:
            embeddings_data_prep_sampling_seed_list = [str(i) for i in range(42, 52)]
        case EmbeddingsDataPrepSamplingSeedListOption.TWENTY_SEEDS:
            embeddings_data_prep_sampling_seed_list = [str(i) for i in range(42, 62)]
        case _:
            msg: str = f"Unknown {args.embeddings_data_prep_sampling_seed_list_option = }"
            raise ValueError(
                msg,
            )

    match args.embeddings_data_prep_num_samples_list_option:
        case EmbeddingsDataPrepNumSamplesListOption.DEFAULT:
            embeddings_data_prep_num_samples_list = [
                "30000",
            ]
        case EmbeddingsDataPrepNumSamplesListOption.SINGLE_CHOICE_50000:
            embeddings_data_prep_num_samples_list = [
                "50000",
            ]
        case EmbeddingsDataPrepNumSamplesListOption.SINGLE_CHOICE_100000:
            embeddings_data_prep_num_samples_list = [
                "100000",
            ]
        case EmbeddingsDataPrepNumSamplesListOption.FIVE_CHOICES_10000_STEPS:
            embeddings_data_prep_num_samples_list = [
                "20000",
                "30000",
                "40000",
                "50000",
                "60000",
            ]
        case EmbeddingsDataPrepNumSamplesListOption.SINGLE_CHOICE_250000:
            embeddings_data_prep_num_samples_list = [
                "250000",
            ]
        case _:
            msg: str = f"Unknown {args.embeddings_data_prep_num_samples_list_option = }"
            raise ValueError(
                msg,
            )

    match args.local_estimates_filtering_num_samples_list:
        case LocalEstimatesFilteringNumSamplesListOption.DEFAULT:
            local_estimates_filtering_num_samples_list = None
        case LocalEstimatesFilteringNumSamplesListOption.FEW_SMALL_STEPS_NUM_SAMPLES:
            local_estimates_filtering_num_samples_list = [
                "2500",
                "5000",
                "7500",
                "10000",
            ]
        case LocalEstimatesFilteringNumSamplesListOption.MEDIUM_SMALL_STEPS_NUM_SAMPLES:
            local_estimates_filtering_num_samples_list = [
                "2500",
                "5000",
                "7500",
                "10000",
                "12500",
                "15000",
            ]
        case LocalEstimatesFilteringNumSamplesListOption.UP_TO_30000_WITH_STEP_SIZE_2500_NUM_SAMPLES:
            local_estimates_filtering_num_samples_list = [str(i * 2500) for i in range(1, 13)]
        case LocalEstimatesFilteringNumSamplesListOption.UP_TO_30000_WITH_STEP_SIZE_5000_NUM_SAMPLES:
            local_estimates_filtering_num_samples_list = [str(i * 5000) for i in range(1, 7)]
        case LocalEstimatesFilteringNumSamplesListOption.UP_TO_50000_WITH_STEP_SIZE_5000_NUM_SAMPLES:
            local_estimates_filtering_num_samples_list = [str(i * 5000) for i in range(1, 11)]
        case LocalEstimatesFilteringNumSamplesListOption.UP_TO_90000_WITH_STEP_SIZE_5000_NUM_SAMPLES:
            local_estimates_filtering_num_samples_list = [str(i * 5000) for i in range(1, 19)]
        case LocalEstimatesFilteringNumSamplesListOption.UP_TO_90000_WITH_STEP_SIZE_10000_NUM_SAMPLES:
            local_estimates_filtering_num_samples_list = [str(i * 10000) for i in range(1, 10)]
        case _:
            msg: str = f"Unknown {args.local_estimates_filtering_num_samples_list = }"
            raise ValueError(msg)

    match args.local_estimates_pointwise_absolute_n_neighbors_list:
        case LocalEstimatesPointwiseAbsoluteNNeighborsListOption.DEFAULT:
            local_estimates_pointwise_absolute_n_neighbors_list = [
                "256",
            ]
        case LocalEstimatesPointwiseAbsoluteNNeighborsListOption.SINGLE_CHOICE_128:
            local_estimates_pointwise_absolute_n_neighbors_list = [
                "128",
            ]
        case LocalEstimatesPointwiseAbsoluteNNeighborsListOption.POWERS_OF_TWO_UP_TO_1024:
            local_estimates_pointwise_absolute_n_neighbors_list = [
                "16",
                "32",
                "64",
                "128",
                "256",
                "512",
                "1024",
            ]
        case _:
            msg: str = f"Unknown {args.local_estimates_pointwise_absolute_n_neighbors_list = }"
            raise ValueError(
                msg,
            )

    # # # #
    # Handle the potential creation of POS tags
    add_prefix_space = False  # Default is False
    additional_data_options = None  # Default is no additional data options

    if args.add_prefix_space:
        # Manually set the add_prefix_space option
        add_prefix_space = True

    if args.create_pos_tags:
        print(  # noqa: T201 - We want this script to print this output
            "create_pos_tags is set to True.",
        )

        add_prefix_space = True
        print(  # noqa: T201 - We want this script to print this output
            f"Setting {add_prefix_space = }, which is required for our POS tagging to work, "
            f"since the tokenizer will be presented with input pre-split into words.",
        )

        additional_data_options = [
            "+data.dataset_type=huggingface_dataset_named_entity",
        ]

    submissions_config = SubmissionConfig(
        add_prefix_space=add_prefix_space,
        submission_mode=args.submission_mode,
        queue=args.queue,
        template=args.template,
        memory=args.memory,
        ncpus=args.ncpus,
        ngpus=args.ngpus,
        walltime=args.walltime,
        additional_overrides=args.additional_overrides,
        data_list=data_list,
        data_number_of_samples_list=data_number_of_samples_list,
        additional_data_options=additional_data_options,
        language_model_list=language_model_list,
        language_model_seed_list=language_model_seed_list,
        checkpoint_no_list=checkpoint_no_list,
        embeddings_data_prep_sampling_mode=args.embeddings_data_prep_sampling_mode,
        embeddings_data_prep_sampling_seed_list=embeddings_data_prep_sampling_seed_list,
        embeddings_data_prep_num_samples_list=embeddings_data_prep_num_samples_list,
        local_estimates_filtering_num_samples_list=local_estimates_filtering_num_samples_list,
        local_estimates_pointwise_absolute_n_neighbors_list=local_estimates_pointwise_absolute_n_neighbors_list,
        finetuning_datasets_list=finetuning_datasets_list,
        finetuning_seed_list=finetuning_seed_list,
        num_train_epochs=num_train_epochs,
        lr_scheduler_type=lr_scheduler_type,
        skip_compute_and_store_embeddings=args.skip_compute_and_store_embeddings,
    )

    run_task(
        submissions_config=submissions_config,
        task=args.task,
        dry_run=args.dry_run,
    )


def main() -> None:
    """Run the submission."""
    args = parse_arguments()

    make_config_and_run_task(
        args=args,
    )


if __name__ == "__main__":
    main()

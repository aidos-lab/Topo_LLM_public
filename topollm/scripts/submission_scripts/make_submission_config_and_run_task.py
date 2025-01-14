# Copyright 2024-2025
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (mail@ruppik.net)
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

import pathlib
import subprocess

from topollm.scripts.submission_scripts.get_checkpoint_no_list import get_checkpoint_no_list
from topollm.scripts.submission_scripts.submission_config import (
    MachineConfig,
    SubmissionConfig,
    Template,
    pick_selected_options_in_each_list,
)
from topollm.scripts.submission_scripts.types import (
    CheckpointNoListOption,
    DataListOption,
    DataSubsamplingNumberOfSamplesListOption,
    DataSubsamplingSamplingSeedListOption,
    EmbeddingsDataPrepNumSamplesListOption,
    EmbeddingsDataPrepSamplingSeedListOption,
    ExperimentSelector,
    ExperimentStage,
    FinetuningBaseModelListOption,
    FinetuningDatasetsListOption,
    FinetuningRegimeOption,
    LanguageModelListOption,
    LocalEstimatesFilteringNumSamplesListOption,
    LocalEstimatesPointwiseAbsoluteNNeighborsListOption,
    ModelGroupOption,
    RunOnlySelectedConfigsOption,
    RunOption,
    SeedListOption,
)
from topollm.typing.enums import (
    DataSamplingMode,
    EmbeddingDataHandlerMode,
    EmbeddingsDataPrepSamplingMode,
    SubmissionMode,
    Task,
)

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
    "wikitext-103-v1_test",
    "wikitext-103-v1_train",
    "wikitext-103-v1_validation",
]
train_split_only_data_list: list[str] = [data_name for data_name in full_data_list if "_train" in data_name]
validation_split_only_data_list: list[str] = [data_name for data_name in full_data_list if "_validation" in data_name]


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


def retrieve_data_list(
    data_list_option: DataListOption,
) -> list[str]:
    """Retrieve the data list based on the option."""
    match data_list_option:
        case DataListOption.FULL:
            data_list = full_data_list
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
        # Selecting only one dataset
        case DataListOption.ICLR_ONLY:
            data_list: list[str] = [
                "iclr_2024_submissions_test",
                "iclr_2024_submissions_train",
                "iclr_2024_submissions_validation",
            ]
        case DataListOption.ICLR_TEST_ONLY:
            data_list: list[str] = [
                "iclr_2024_submissions_test",
            ]
        case DataListOption.ICLR_TRAIN_ONLY:
            data_list: list[str] = [
                "iclr_2024_submissions_train",
            ]
        case DataListOption.ICLR_VALIDATION_ONLY:
            data_list: list[str] = [
                "iclr_2024_submissions_validation",
            ]
        case DataListOption.MULTIWOZ21_ONLY:
            data_list: list[str] = [
                "multiwoz21_test",
                "multiwoz21_train",
                "multiwoz21_validation",
            ]
        case DataListOption.MULTIWOZ21_TEST_ONLY:
            data_list: list[str] = [
                "multiwoz21_test",
            ]
        case DataListOption.MULTIWOZ21_TRAIN_ONLY:
            data_list: list[str] = [
                "multiwoz21_train",
            ]
        case DataListOption.MULTIWOZ21_VALIDATION_ONLY:
            data_list: list[str] = [
                "multiwoz21_validation",
            ]
        case DataListOption.REDDIT_ONLY:
            data_list: list[str] = [
                "one-year-of-tsla-on-reddit_test",
                "one-year-of-tsla-on-reddit_train",
                "one-year-of-tsla-on-reddit_validation",
            ]
        case DataListOption.REDDIT_TEST_ONLY:
            data_list: list[str] = [
                "one-year-of-tsla-on-reddit_test",
            ]
        case DataListOption.REDDIT_TRAIN_ONLY:
            data_list: list[str] = [
                "one-year-of-tsla-on-reddit_train",
            ]
        case DataListOption.REDDIT_VALIDATION_ONLY:
            data_list: list[str] = [
                "one-year-of-tsla-on-reddit_validation",
            ]
        case DataListOption.SGD_ONLY:
            data_list: list[str] = [
                "sgd_test",
                "sgd_train",
                "sgd_validation",
            ]
        case DataListOption.SGD_TEST_ONLY:
            data_list: list[str] = [
                "sgd_test",
            ]
        case DataListOption.SGD_TRAIN_ONLY:
            data_list: list[str] = [
                "sgd_train",
            ]
        case DataListOption.SGD_VALIDATION_ONLY:
            data_list: list[str] = [
                "sgd_validation",
            ]
        case DataListOption.WIKITEXT_ONLY:
            data_list: list[str] = [
                "wikitext-103-v1_test",
                "wikitext-103-v1_train",
                "wikitext-103-v1_validation",
            ]
        case DataListOption.WIKITEXT_TEST_ONLY:
            data_list: list[str] = [
                "wikitext-103-v1_test",
            ]
        case DataListOption.WIKITEXT_TRAIN_ONLY:
            data_list: list[str] = [
                "wikitext-103-v1_train",
            ]
        case DataListOption.WIKITEXT_VALIDATION_ONLY:
            data_list: list[str] = [
                "wikitext-103-v1_validation",
            ]
        # Select certain data splits
        case DataListOption.TRAIN_SPLIT_ONLY:
            data_list = train_split_only_data_list
        case DataListOption.VALIDATION_SPLIT_ONLY:
            data_list = validation_split_only_data_list
        # Mixing two datasets
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
        case DataListOption.MULTIWOZ21_VALIDATION_AND_REDDIT_VALIDATION:
            data_list: list[str] = [
                "multiwoz21_validation",
                "one-year-of-tsla-on-reddit_validation",
            ]
        # Mixing three datasets
        case DataListOption.ICLR_VALIDATION_AND_SGD_VALIDATION_AND_WIKITEXT_VALIDATION:
            data_list: list[str] = [
                "iclr_2024_submissions_validation",
                "sgd_validation",
                "wikitext-103-v1_validation",
            ]
        case _:
            msg = f"Unknown {data_list_option = }"
            raise ValueError(
                msg,
            )

    return data_list


def retrieve_data_subsampling_number_of_samples_list(
    data_subsampling_number_of_samples_list_option: DataSubsamplingNumberOfSamplesListOption,
) -> list[str] | None:
    """Retrieve the data subsampling number of samples list based on the option."""
    match data_subsampling_number_of_samples_list_option:
        case DataSubsamplingNumberOfSamplesListOption.NONE:
            data_subsampling_number_of_samples_list = None
        case DataSubsamplingNumberOfSamplesListOption.FIXED_3000:
            data_subsampling_number_of_samples_list = [
                "3000",
            ]
        case DataSubsamplingNumberOfSamplesListOption.FIXED_10000:
            data_subsampling_number_of_samples_list = [
                "10000",
            ]
        case DataSubsamplingNumberOfSamplesListOption.FIXED_12000:
            data_subsampling_number_of_samples_list = [
                "12000",
            ]
        case DataSubsamplingNumberOfSamplesListOption.FIXED_16000:
            data_subsampling_number_of_samples_list = [
                "16000",
            ]
        case DataSubsamplingNumberOfSamplesListOption.FIXED_22000:
            data_subsampling_number_of_samples_list = [
                "22000",
            ]
        case DataSubsamplingNumberOfSamplesListOption.RANGE_START_2000_STOP_12000_STEP_2000:
            data_subsampling_number_of_samples_list = [
                str(i)
                for i in range(
                    2_000,
                    12_000,
                    2_000,
                )
            ]
        case DataSubsamplingNumberOfSamplesListOption.RANGE_START_2000_STOP_18000_STEP_2000:
            data_subsampling_number_of_samples_list = [
                str(i)
                for i in range(
                    2_000,
                    18_000,
                    2_000,
                )
            ]
        case DataSubsamplingNumberOfSamplesListOption.RANGE_START_12000_STOP_18000_STEP_2000:
            data_subsampling_number_of_samples_list = [
                str(i)
                for i in range(
                    12_000,
                    18_000,
                    2_000,
                )
            ]
        case DataSubsamplingNumberOfSamplesListOption.RANGE_START_2000_STOP_24000_STEP_2000:
            data_subsampling_number_of_samples_list = [
                str(i)
                for i in range(
                    2_000,
                    24_000,
                    2_000,
                )
            ]
        case DataSubsamplingNumberOfSamplesListOption.RANGE_START_12000_STOP_24000_STEP_2000:
            data_subsampling_number_of_samples_list = [
                str(i)
                for i in range(
                    12_000,
                    24_000,
                    2_000,
                )
            ]
        case _:
            msg = f"Unknown {data_subsampling_number_of_samples_list_option = }"
            raise ValueError(
                msg,
            )

    return data_subsampling_number_of_samples_list


def retrieve_data_subsampling_sampling_seed_list(
    data_subsampling_sampling_seed_list_option: DataSubsamplingSamplingSeedListOption,
) -> list[str] | None:
    match data_subsampling_sampling_seed_list_option:
        case DataSubsamplingSamplingSeedListOption.NONE:
            data_subsampling_sampling_seed_list = None
        case DataSubsamplingSamplingSeedListOption.DEFAULT:
            data_subsampling_sampling_seed_list = [
                "778",
            ]
        case DataSubsamplingSamplingSeedListOption.FIXED_777:
            data_subsampling_sampling_seed_list = [
                "777",
            ]
        case DataSubsamplingSamplingSeedListOption.TWO_SEEDS:
            data_subsampling_sampling_seed_list = [
                "778",
                "779",
            ]
        case DataSubsamplingSamplingSeedListOption.THREE_SEEDS:
            data_subsampling_sampling_seed_list = [
                "778",
                "779",
                "780",
            ]
        case DataSubsamplingSamplingSeedListOption.FIVE_SEEDS:
            data_subsampling_sampling_seed_list = [str(i) for i in range(778, 783)]
        case DataSubsamplingSamplingSeedListOption.TEN_SEEDS:
            data_subsampling_sampling_seed_list = [str(i) for i in range(778, 788)]
        case DataSubsamplingSamplingSeedListOption.TWENTY_SEEDS:
            data_subsampling_sampling_seed_list = [str(i) for i in range(778, 798)]
        case _:
            msg = f"Unknown {data_subsampling_sampling_seed_list_option = }"
            raise ValueError(
                msg,
            )

    return data_subsampling_sampling_seed_list


def retrieve_finetuning_base_model_list(
    finetuning_base_model_list_option: FinetuningBaseModelListOption,
) -> list[str]:
    """Retrieve the finetuning base model list based on the option."""
    match finetuning_base_model_list_option:
        case FinetuningBaseModelListOption.ROBERTA_BASE:
            finetuning_base_model_list: list[str] = [
                "roberta-base_for_masked_lm",
            ]
        case FinetuningBaseModelListOption.GPT2_MEDIUM:
            finetuning_base_model_list: list[str] = [
                "gpt2-medium_for_causal_lm",
            ]
        case _:
            msg = f"Unknown {finetuning_base_model_list_option = }"
            raise ValueError(
                msg,
            )

    return finetuning_base_model_list


def retrieve_finetuning_datasets_list(
    finetuning_datasets_list_option: FinetuningDatasetsListOption,
) -> list[str]:
    """Retrieve the finetuning datasets list based on the option."""
    match finetuning_datasets_list_option:
        case FinetuningDatasetsListOption.DEBUG:
            finetuning_datasets_list: list[str] = [
                "train_and_eval_on_multiwoz21_train-samples-small",
            ]
        case FinetuningDatasetsListOption.MANUAL_IN_PYTHON_SCRIPT:
            finetuning_datasets_list: list[str] = [
                "train_and_eval_on_multiwoz21_train-samples-small",
                "train_and_eval_on_one-year-of-tsla-on-reddit_train-samples-small",
            ]
        case FinetuningDatasetsListOption.ICLR_SMALL:
            finetuning_datasets_list: list[str] = [
                "train_and_eval_on_iclr_2024_submissions_train-samples-5000",
            ]
        case FinetuningDatasetsListOption.MULTIWOZ21_SMALL:
            finetuning_datasets_list: list[str] = [
                "train_and_eval_on_multiwoz21_train-samples-small",
            ]
        case FinetuningDatasetsListOption.MULTIWOZ21_FULL:
            finetuning_datasets_list: list[str] = [
                "train_and_eval_on_multiwoz21_train-samples-full",
            ]
        case FinetuningDatasetsListOption.REDDIT_SMALL:
            finetuning_datasets_list: list[str] = [
                "train_and_eval_on_one-year-of-tsla-on-reddit_train-samples-small",
            ]
        case FinetuningDatasetsListOption.REDDIT_FULL:
            finetuning_datasets_list: list[str] = [
                "train_and_eval_on_one-year-of-tsla-on-reddit_train-samples-full",
            ]
        case FinetuningDatasetsListOption.SGD_SMALL:
            finetuning_datasets_list: list[str] = [
                "train_and_eval_on_sgd_train-samples-small",
            ]
        case FinetuningDatasetsListOption.WIKITEXT_SMALL:
            finetuning_datasets_list: list[str] = [
                "train_and_eval_on_wikitext_train-samples-small",
            ]
        case FinetuningDatasetsListOption.MULTIWOZ21_AND_REDDIT_SMALL:
            finetuning_datasets_list: list[str] = [
                "train_and_eval_on_multiwoz21_train-samples-small",
                "train_and_eval_on_one-year-of-tsla-on-reddit_train-samples-small",
            ]
        case FinetuningDatasetsListOption.MULTIWOZ21_AND_REDDIT_FULL:
            finetuning_datasets_list: list[str] = [
                "train_and_eval_on_multiwoz21_train-samples-full",
                "train_and_eval_on_one-year-of-tsla-on-reddit_train-samples-full",
            ]
        case _:
            msg = f"Unknown {finetuning_datasets_list_option = }"
            raise ValueError(
                msg,
            )

    return finetuning_datasets_list


def retrieve_local_estimates_pointwise_absolute_n_neighbors_list(
    local_estimates_pointwise_absolute_n_neighbors_list_option: LocalEstimatesPointwiseAbsoluteNNeighborsListOption,
) -> list[str]:
    """Retrieve the local estimates pointwise absolute n neighbors list based on the option."""
    match local_estimates_pointwise_absolute_n_neighbors_list_option:
        case LocalEstimatesPointwiseAbsoluteNNeighborsListOption.DEFAULT:
            local_estimates_pointwise_absolute_n_neighbors_list = [
                "128",
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
            msg: str = f"Unknown {local_estimates_pointwise_absolute_n_neighbors_list_option = }"
            raise ValueError(
                msg,
            )

    return local_estimates_pointwise_absolute_n_neighbors_list


def retrieve_model_and_checkpoint_list(
    language_model_list_option: LanguageModelListOption,
    checkpoint_no_list_option: CheckpointNoListOption,
    num_train_epochs: str,
) -> tuple[
    list[str],
    list[str] | None,
]:
    """Retrieve the language model list and checkpoint number list based on the option."""
    match language_model_list_option:
        # # # #
        # RoBERTa-base models
        case LanguageModelListOption.ROBERTA_BASE:
            language_model_list: list[str] = [
                "roberta-base",
            ]
            # No checkpoints for the unmodified model
            checkpoint_no_list = None
        case LanguageModelListOption.FINETUNED_ON_OLD_AND_NEW_DATA_FEW_EPOCHS_FROM_ROBERTA_BASE:
            language_model_list: list[str] = [
                "roberta-base-masked_lm-defaults_one-year-of-tsla-on-reddit-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5",
                "roberta-base-masked_lm-defaults_wikitext-103-v1-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5",
            ]

            checkpoint_no_list = get_checkpoint_no_list(
                checkpoint_no_list_option=checkpoint_no_list_option,
                num_train_epochs=int(num_train_epochs),
            )
        case LanguageModelListOption.FINETUNED_ON_OLD_AND_NEW_DATA_FEW_EPOCHS_FROM_GPT2_MEDIUM:
            language_model_list: list[str] = [
                "gpt2-medium-causal_lm-defaults_one-year-of-tsla-on-reddit-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5",
                "gpt2-medium-causal_lm-defaults_wikitext-103-v1-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5",
            ]

            checkpoint_no_list = get_checkpoint_no_list(
                checkpoint_no_list_option=checkpoint_no_list_option,
                num_train_epochs=int(num_train_epochs),
            )
        case LanguageModelListOption.FINETUNED_ON_MULTIWOZ_DATA_FEW_EPOCHS_FROM_ROBERTA_BASE:
            language_model_list: list[str] = [
                "roberta-base-masked_lm-defaults_multiwoz21-rm-empty-True-do_nothing-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5",
            ]

            checkpoint_no_list = get_checkpoint_no_list(
                checkpoint_no_list_option=checkpoint_no_list_option,
                num_train_epochs=int(num_train_epochs),
            )
        case LanguageModelListOption.FINETUNED_ON_MULTIWOZ_DATA_FEW_EPOCHS_FROM_GPT2_MEDIUM:
            language_model_list: list[str] = [
                "gpt2-medium-causal_lm-defaults_multiwoz21-rm-empty-True-do_nothing-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5",
            ]

            checkpoint_no_list = get_checkpoint_no_list(
                checkpoint_no_list_option=checkpoint_no_list_option,
                num_train_epochs=int(num_train_epochs),
            )
        case LanguageModelListOption.FINETUNED_ON_MULTIWOZ_AND_REDDIT_AND_WIKITEXT_DATA_FEW_EPOCHS_FROM_ROBERTA_BASE:
            language_model_list: list[str] = [
                "roberta-base-masked_lm-defaults_multiwoz21-rm-empty-True-do_nothing-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5",
                "roberta-base-masked_lm-defaults_one-year-of-tsla-on-reddit-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5",
                "roberta-base-masked_lm-defaults_wikitext-103-v1-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5",
            ]

            checkpoint_no_list = get_checkpoint_no_list(
                checkpoint_no_list_option=checkpoint_no_list_option,
                num_train_epochs=int(num_train_epochs),
            )
        case LanguageModelListOption.FINETUNED_ON_MULTIWOZ_AND_REDDIT_AND_WIKITEXT_DATA_FEW_EPOCHS_FROM_GPT2_MEDIUM:
            language_model_list: list[str] = [
                "gpt2-medium-causal_lm-defaults_multiwoz21-rm-empty-True-do_nothing-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5",
                "gpt2-medium-causal_lm-defaults_one-year-of-tsla-on-reddit-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5",
                "gpt2-medium-causal_lm-defaults_wikitext-103-v1-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5",
            ]

            checkpoint_no_list = get_checkpoint_no_list(
                checkpoint_no_list_option=checkpoint_no_list_option,
                num_train_epochs=int(num_train_epochs),
            )
        case LanguageModelListOption.SELECTED_FINETUNED_FEW_EPOCHS_FROM_ROBERTA_BASE:
            language_model_list: list[str] = [
                "roberta-base-masked_lm-defaults_multiwoz21-rm-empty-True-do_nothing-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5",
                "roberta-base-masked_lm-defaults_one-year-of-tsla-on-reddit-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5",
            ]

            checkpoint_no_list = get_checkpoint_no_list(
                checkpoint_no_list_option=checkpoint_no_list_option,
                num_train_epochs=int(num_train_epochs),
            )
        case LanguageModelListOption.FULL_FINETUNED_FEW_EPOCHS_FROM_ROBERTA_BASE:
            language_model_list: list[str] = [
                "roberta-base-masked_lm-defaults_iclr_2024_submissions-rm-empty-True-do_nothing-ner_tags_train-5000-take_first-111_standard-None_5e-05-linear-0.01-5"
                "roberta-base-masked_lm-defaults_multiwoz21-rm-empty-True-do_nothing-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5",
                "roberta-base-masked_lm-defaults_one-year-of-tsla-on-reddit-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5",
                "roberta-base-masked_lm-defaults_wikitext-103-v1-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5",
            ]

            checkpoint_no_list = get_checkpoint_no_list(
                checkpoint_no_list_option=checkpoint_no_list_option,
                num_train_epochs=int(num_train_epochs),
            )
        case LanguageModelListOption.SELECTED_FINETUNED_MANY_EPOCHS_FROM_ROBERTA_BASE:
            language_model_list: list[str] = [
                "model-roberta-base_task-masked_lm_multiwoz21-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50",
                "model-roberta-base_task-masked_lm_one-year-of-tsla-on-reddit-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50",
            ]

            checkpoint_no_list = get_checkpoint_no_list(
                checkpoint_no_list_option=checkpoint_no_list_option,
                num_train_epochs=int(num_train_epochs),
            )
        case LanguageModelListOption.WITH_005_015_02_DROPOUT_FINETUNED_ON_MULTIWOZ_SMALL_MANY_EPOCHS_FROM_ROBERTA_BASE:
            language_model_list: list[str] = [
                "roberta-base-masked_lm-0.05-0.05-None_multiwoz21-do_nothing-ner_tags_train-10000-take_first-111_standard-None_5e-05-constant-0.01-50.yaml",
                "roberta-base-masked_lm-0.15-0.15-None_multiwoz21-do_nothing-ner_tags_train-10000-take_first-111_standard-None_5e-05-constant-0.01-50.yaml",
                "roberta-base-masked_lm-0.2-0.2-None_multiwoz21-do_nothing-ner_tags_train-10000-take_first-111_standard-None_5e-05-constant-0.01-50",
            ]

            checkpoint_no_list = get_checkpoint_no_list(
                checkpoint_no_list_option=checkpoint_no_list_option,
                num_train_epochs=int(num_train_epochs),
            )
        case LanguageModelListOption.WITH_006_007_DROPOUT_FINETUNED_ON_MULTIWOZ_SMALL_MANY_EPOCHS_FROM_ROBERTA_BASE:
            language_model_list: list[str] = [
                "roberta-base-masked_lm-0.06-0.06-None_multiwoz21-do_nothing-ner_tags_train-10000-take_first-111_standard-None_5e-05-constant-0.01-50.yaml",
                "roberta-base-masked_lm-0.07-0.07-None_multiwoz21-do_nothing-ner_tags_train-10000-take_first-111_standard-None_5e-05-constant-0.01-50.yaml",
            ]

            checkpoint_no_list = get_checkpoint_no_list(
                checkpoint_no_list_option=checkpoint_no_list_option,
                num_train_epochs=int(num_train_epochs),
            )
        # # # #
        # GPT-2 models
        case LanguageModelListOption.GPT2_MEDIUM:
            language_model_list: list[str] = [
                "gpt2-medium",
            ]
            # No checkpoints for the unmodified model
            checkpoint_no_list = None
        # # # #
        # Other models
        case LanguageModelListOption.SETSUMBT_SELECTED:
            setsumbt_seed: int = 0
            language_model_list: list[str] = setsumbt_model_list

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
                msg: str = f"Unknown {setsumbt_seed = }"
                raise ValueError(msg)
        case _:
            msg: str = f"Unknown {language_model_list_option = }"
            raise ValueError(
                msg,
            )

    return language_model_list, checkpoint_no_list


def retrieve_language_model_seed_list(
    language_model_seed_list_option: SeedListOption,
) -> list[str] | None:
    """Retrieve the language model seed list based on the option."""
    match language_model_seed_list_option:
        case SeedListOption.DO_NOT_SET:
            language_model_seed_list = None
        case SeedListOption.ONE_SEED:
            language_model_seed_list = seed_list_option_one_seed
        case SeedListOption.TWO_SEEDS:
            language_model_seed_list = seed_list_option_two_seeds
        case SeedListOption.FIVE_SEEDS:
            language_model_seed_list = seed_list_option_five_seeds
        case SeedListOption.FIXED_SEED_1234:
            language_model_seed_list = [
                "1234",
            ]
        case SeedListOption.FIXED_SEEDS_1235_1236:
            language_model_seed_list = [
                "1235",
                "1236",
            ]
        case SeedListOption.FIXED_SEEDS_1234_1235_1236:
            language_model_seed_list = [
                "1234",
                "1235",
                "1236",
            ]
        case _:
            msg: str = f"Unknown {language_model_seed_list_option = }"
            raise ValueError(
                msg,
            )

    return language_model_seed_list


def retrieve_finetuning_seed_list(
    finetuning_seed_list_option: SeedListOption,
) -> list[str] | None:
    """Retrieve the finetuning seed list based on the option."""
    match finetuning_seed_list_option:
        case SeedListOption.DO_NOT_SET:
            finetuning_seed_list = None
        case SeedListOption.ONE_SEED:
            finetuning_seed_list = seed_list_option_one_seed
        case SeedListOption.TWO_SEEDS:
            finetuning_seed_list = seed_list_option_two_seeds
        case SeedListOption.FIVE_SEEDS:
            finetuning_seed_list = seed_list_option_five_seeds
        case _:
            msg: str = f"Unknown {finetuning_seed_list_option = }"
            raise ValueError(
                msg,
            )

    return finetuning_seed_list


def retrieve_embeddings_data_prep_sampling_seed_list(
    embeddings_data_prep_sampling_seed_list_option: EmbeddingsDataPrepSamplingSeedListOption,
) -> list[str]:
    """Retrieve the embeddings data prep sampling seed list based on the option."""
    match embeddings_data_prep_sampling_seed_list_option:
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
            msg: str = f"Unknown {embeddings_data_prep_sampling_seed_list_option = }"
            raise ValueError(
                msg,
            )

    return embeddings_data_prep_sampling_seed_list


def retrieve_embeddings_data_prep_num_samples_list(
    embeddings_data_prep_num_samples_list_option: EmbeddingsDataPrepNumSamplesListOption,
) -> list[str]:
    """Retrieve the embeddings data prep number of samples list based on the option."""
    match embeddings_data_prep_num_samples_list_option:
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
        case EmbeddingsDataPrepNumSamplesListOption.SINGLE_CHOICE_150000:
            embeddings_data_prep_num_samples_list = [
                "150000",
            ]
        case EmbeddingsDataPrepNumSamplesListOption.SINGLE_CHOICE_250000:
            embeddings_data_prep_num_samples_list = [
                "250000",
            ]
        case EmbeddingsDataPrepNumSamplesListOption.FIVE_CHOICES_10000_STEPS:
            embeddings_data_prep_num_samples_list = [
                "20000",
                "30000",
                "40000",
                "50000",
                "60000",
            ]
        case _:
            msg: str = f"Unknown {embeddings_data_prep_num_samples_list_option = }"
            raise ValueError(
                msg,
            )

    return embeddings_data_prep_num_samples_list


def retrieve_local_estimates_filtering_num_samples_list(
    local_estimates_filtering_num_samples_list_option: LocalEstimatesFilteringNumSamplesListOption,
) -> list[str]:
    """Retrieve the local estimates filtering number of samples list based on the option."""
    match local_estimates_filtering_num_samples_list_option:
        case (
            LocalEstimatesFilteringNumSamplesListOption.DEFAULT
            | LocalEstimatesFilteringNumSamplesListOption.SINGLE_CHOICE_60000
        ):
            local_estimates_filtering_num_samples_list: list[str] = [
                "60000",
            ]
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
        case LocalEstimatesFilteringNumSamplesListOption.RANGE_START_20000_STOP_110000_STEP_20000:
            local_estimates_filtering_num_samples_list = [
                str(object=i)
                for i in range(
                    20_000,
                    110_000,
                    20_000,
                )
            ]
        case LocalEstimatesFilteringNumSamplesListOption.RANGE_START_10000_STOP_110000_STEP_10000:
            local_estimates_filtering_num_samples_list = [
                str(object=i)
                for i in range(
                    10_000,
                    110_000,
                    10_000,
                )
            ]
        case _:
            msg: str = f"Unknown {local_estimates_filtering_num_samples_list_option = }"
            raise ValueError(
                msg,
            )

    return local_estimates_filtering_num_samples_list


def make_machine_config(
    task: Task,
    model_group_option: ModelGroupOption,
    experiment_stage: ExperimentStage,
    experiment_selector: ExperimentSelector,
    template_to_use_for_compute_embeddings: Template,
    embedding_data_handler_mode: EmbeddingDataHandlerMode,
    memory: str,
    ncpus: str,
    ngpus: str,
    queue: str,
    template: Template,
) -> MachineConfig:
    """Make a machine configuration for the experiment.

    This function encapsulates all the logic for selecting the correct machine configuration for the experiment.

    Default resource configurations
    -------------------------------
    - 16GB of memory is not enough for the embeddings data prep step
      for dataset subsampling sample size 10_000 on the multiwoz21_train and reddit_train datasets.
    - 32GB of memory is enough for the embeddings data prep step
      for dataset subsampling sample size 12_000 on the multiwoz21_train and reddit_train datasets.
    """
    # Default walltime
    walltime = "08:00:00"

    match experiment_stage:
        case ExperimentStage.COMPUTE_EMBEDDINGS_PLUS_SINGLE_PIPELINE_RUN:
            match template_to_use_for_compute_embeddings:
                case Template.RTX6000:
                    queue, template = "CUDA", Template.RTX6000
                case Template.GTX1080:
                    queue, template = "CUDA", Template.GTX1080
                case Template.DSML:
                    queue, template = "DSML", Template.DSML
                case _:
                    msg: str = f"{template_to_use_for_compute_embeddings = } requires a GPU template."
                    raise ValueError(
                        msg,
                    )

            ncpus = "4"
            ngpus = "1"

            match embedding_data_handler_mode:
                case EmbeddingDataHandlerMode.REGULAR:
                    # For the datasets with 10_000 samples,
                    # one pipeline run with regular embeddings usually takes about 30 min.
                    # We set the walltime to 2 hours to be on the safe side.
                    walltime = "02:00:00"
                case EmbeddingDataHandlerMode.MASKED_TOKEN:
                    # For the masked embeddings on datasets with long sequences,
                    # the walltime can be significantly longer.
                    walltime = "06:00:00"  # Longer walltime to make sure it is long enough for the masked embeddings
                case _:
                    msg: str = f"Unknown {embedding_data_handler_mode = }"
                    raise ValueError(
                        msg,
                    )

            match model_group_option:
                case (
                    ModelGroupOption.ROBERTA_BASE_WITHOUT_MODIFICATIONS
                    | ModelGroupOption.ROBERTA_BASE_FINETUNED_FOR_FEW_EPOCHS_OLD_AND_NEW_DATA_SINGLE_SEED_LAST_CHECKPOINT
                    | ModelGroupOption.ROBERTA_BASE_FINETUNED_FOR_FEW_EPOCHS_MULTIWOZ_DATA_SINGLE_SEED_LAST_CHECKPOINT
                    | ModelGroupOption.ROBERTA_BASE_FINETUNED_FOR_FEW_EPOCHS_MULTIWOZ_AND_REDDIT_AND_WIKITEXT_DATA_SINGLE_SEED_ALL_CHECKPOINTS
                    | ModelGroupOption.ROBERTA_BASE_FINETUNED_FOR_MANY_EPOCHS
                ):
                    # For the RoBERTa base model, we need less memory since the embeddings have lower dimensionality.
                    memory = "32"
                case (
                    ModelGroupOption.GPT2_MEDIUM_WITHOUT_MODIFICATIONS
                    | ModelGroupOption.GPT2_MEDIUM_FINETUNED_FOR_FEW_EPOCHS_MULTIWOZ_AND_REDDIT_AND_WIKITEXT_DATA_SINGLE_SEED_LAST_CHECKPOINT
                ):
                    # For the GPT2 medium model, we need more memory since the embeddings have higher dimensionality.
                    # The embeddings data prep step failed with 32GB of memory for the GPT2 medium model.
                    memory = "64"
                case _:
                    msg: str = f"The {model_group_option = } is not yet handled in the machine configuration selection."
                    raise ValueError(
                        msg,
                    )

        case (
            ExperimentStage.SKIP_COMPUTE_EMBEDDINGS_BUT_DO_MULTIPLE_PIPELINE_RUNS
            | ExperimentStage.SKIP_COMPUTE_EMBEDDINGS_AND_SKIP_EMBEDDINGS_DATA_PREP
        ):
            ncpus = "6"
            ngpus = "0"
            queue = "DEFAULT"
            template = Template.CPU

            # We set a shorter walltime for the runs which do not compute the embeddings
            walltime = "03:00:00"
        case _:
            msg: str = f"Unknown {experiment_stage = }"
            raise ValueError(
                msg,
            )

    # Override the machine configuration based on the experiment selector
    match experiment_selector:
        case ExperimentSelector.SENSITIVITY_ANALYSIS_MULTIWOZ21_DIFFERENT_DATA_SUBSAMPLING_NUMBER_OF_SAMPLES:
            # Note: We explicitly increase the memory size here,
            # since for the embeddings data prep step on 12_000 and more data subsamlping samples,
            # the embeddings data prep step requires more memory.
            memory = "64"
        case ExperimentSelector.SENSITIVITY_ANALYSIS_REDDIT_DIFFERENT_DATA_SUBSAMPLING_NUMBER_OF_SAMPLES:
            # Note: We explicitly increase the memory size here,
            # since for the embeddings data prep step on 12_000 and more data subsamlping samples,
            # the embeddings data prep step requires more memory.
            memory = "80"

    # Override the machine configuration based on the task
    match task:
        case Task.PERPLEXITY:
            queue = "CUDA"
            template = Template.RTX6000

            walltime = "12:00:00"  # Use slightly longer walltime for perplexity
        case Task.FINETUNING:
            queue = "CUDA"
            template = Template.RTX6000

            ncpus = "4"
            ngpus = "1"
            walltime = "48:00:00"  # Use significantly longer walltime for finetuning

    machine_config = MachineConfig(
        queue=queue,
        template=template,
        memory=memory,
        ncpus=ncpus,
        ngpus=ngpus,
        walltime=walltime,
    )

    return machine_config


def make_submission_config_and_run_task(
    data_list_option: DataListOption,
    data_subsampling_sampling_mode: DataSamplingMode,
    data_subsampling_number_of_samples_list_option: DataSubsamplingNumberOfSamplesListOption,
    data_subsampling_sampling_seed_list_option: DataSubsamplingSamplingSeedListOption,
    fp16: str,
    finetuning_base_model_list_option: FinetuningBaseModelListOption,
    finetuning_datasets_list_option: FinetuningDatasetsListOption,
    finetuning_regime_option: FinetuningRegimeOption,
    finetuning_seed_list_option: SeedListOption,
    batch_size_train: int,
    batch_size_eval: int,
    wandb_project: str,
    embedding_data_handler_mode: EmbeddingDataHandlerMode,
    language_model_list_option: LanguageModelListOption,
    language_model_seed_list_option: SeedListOption,
    checkpoint_no_list_option: CheckpointNoListOption,
    layer_indices_list: list[str],
    embeddings_data_prep_sampling_mode: EmbeddingsDataPrepSamplingMode,
    embeddings_data_prep_sampling_seed_list_option: EmbeddingsDataPrepSamplingSeedListOption,
    embeddings_data_prep_num_samples_list_option: EmbeddingsDataPrepNumSamplesListOption,
    local_estimates_filtering_num_samples_list_option: LocalEstimatesFilteringNumSamplesListOption,
    local_estimates_pointwise_absolute_n_neighbors_list_option: LocalEstimatesPointwiseAbsoluteNNeighborsListOption,
    machine_config: MachineConfig,
    submission_mode: SubmissionMode,
    task: Task,
    additional_overrides: list[str] | None,
    *,
    add_prefix_space: bool,
    create_pos_tags: bool,
    skip_compute_and_store_embeddings_in_pipeline: bool,
    skip_embeddings_data_prep_in_pipeline: bool,
    run_option: RunOption = RunOption.DO_SUBMISSION,
    run_only_selected_configs_option: RunOnlySelectedConfigsOption = RunOnlySelectedConfigsOption.RUN_ALL,
) -> None:
    """Make a submission configuration and run the task."""
    data_list: list[str] = retrieve_data_list(
        data_list_option=data_list_option,
    )

    data_subsampling_number_of_samples_list: list[str] | None = retrieve_data_subsampling_number_of_samples_list(
        data_subsampling_number_of_samples_list_option=data_subsampling_number_of_samples_list_option,
    )

    data_subsampling_sampling_seed_list: list[str] | None = retrieve_data_subsampling_sampling_seed_list(
        data_subsampling_sampling_seed_list_option=data_subsampling_sampling_seed_list_option,
    )

    match finetuning_regime_option:
        case FinetuningRegimeOption.FEW_EPOCHS:
            num_train_epochs = "5"
            lr_scheduler_type = "linear"
        case FinetuningRegimeOption.MANY_EPOCHS_WITH_OVERFITTING_RISK:
            num_train_epochs = "50"
            lr_scheduler_type = "constant"
        case _:
            msg: str = f"Unknown {finetuning_regime_option = }"
            raise ValueError(
                msg,
            )

    (
        language_model_list,
        checkpoint_no_list,
    ) = retrieve_model_and_checkpoint_list(
        language_model_list_option=language_model_list_option,
        checkpoint_no_list_option=checkpoint_no_list_option,
        num_train_epochs=num_train_epochs,
    )

    language_model_seed_list: None | list[str] = retrieve_language_model_seed_list(
        language_model_seed_list_option=language_model_seed_list_option,
    )

    finetuning_base_model_list: list[str] = retrieve_finetuning_base_model_list(
        finetuning_base_model_list_option=finetuning_base_model_list_option,
    )

    finetuning_datasets_list: list[str] = retrieve_finetuning_datasets_list(
        finetuning_datasets_list_option=finetuning_datasets_list_option,
    )

    finetuning_seed_list: list[str] | None = retrieve_finetuning_seed_list(
        finetuning_seed_list_option=finetuning_seed_list_option,
    )

    embeddings_data_prep_sampling_seed_list: list[str] = retrieve_embeddings_data_prep_sampling_seed_list(
        embeddings_data_prep_sampling_seed_list_option=embeddings_data_prep_sampling_seed_list_option,
    )

    embeddings_data_prep_num_samples_list: list[str] = retrieve_embeddings_data_prep_num_samples_list(
        embeddings_data_prep_num_samples_list_option=embeddings_data_prep_num_samples_list_option,
    )

    local_estimates_filtering_num_samples_list: list[str] = retrieve_local_estimates_filtering_num_samples_list(
        local_estimates_filtering_num_samples_list_option=local_estimates_filtering_num_samples_list_option,
    )

    local_estimates_pointwise_absolute_n_neighbors_list: list[str] = (
        retrieve_local_estimates_pointwise_absolute_n_neighbors_list(
            local_estimates_pointwise_absolute_n_neighbors_list_option=local_estimates_pointwise_absolute_n_neighbors_list_option
        )
    )

    additional_data_options = None  # Default is no additional data options

    if create_pos_tags:
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

    print(  # noqa: T201 - we want this function to print
        f"{additional_overrides = }",
    )

    submission_config = SubmissionConfig(
        add_prefix_space=add_prefix_space,
        submission_mode=submission_mode,
        machine_config=machine_config,
        additional_overrides=additional_overrides,
        data_list=data_list,
        data_subsampling_sampling_mode=data_subsampling_sampling_mode,
        data_subsampling_number_of_samples_list=data_subsampling_number_of_samples_list,
        data_subsampling_sampling_seed_list=data_subsampling_sampling_seed_list,
        additional_data_options=additional_data_options,
        embedding_data_handler_mode=embedding_data_handler_mode,
        language_model_list=language_model_list,
        language_model_seed_list=language_model_seed_list,
        checkpoint_no_list=checkpoint_no_list,
        layer_indices_list=layer_indices_list,
        finetuning_base_model_list=finetuning_base_model_list,
        finetuning_fp16=fp16,
        finetuning_datasets_list=finetuning_datasets_list,
        finetuning_seed_list=finetuning_seed_list,
        embeddings_data_prep_sampling_mode=embeddings_data_prep_sampling_mode,
        embeddings_data_prep_sampling_seed_list=embeddings_data_prep_sampling_seed_list,
        embeddings_data_prep_num_samples_list=embeddings_data_prep_num_samples_list,
        local_estimates_filtering_num_samples_list=local_estimates_filtering_num_samples_list,
        local_estimates_pointwise_absolute_n_neighbors_list=local_estimates_pointwise_absolute_n_neighbors_list,
        batch_size_train=batch_size_train,
        batch_size_eval=batch_size_eval,
        num_train_epochs=num_train_epochs,
        lr_scheduler_type=lr_scheduler_type,
        wandb_project=wandb_project,
        skip_compute_and_store_embeddings_in_pipeline=skip_compute_and_store_embeddings_in_pipeline,
        skip_embeddings_data_prep_in_pipeline=skip_embeddings_data_prep_in_pipeline,
    )

    match run_only_selected_configs_option:
        case RunOnlySelectedConfigsOption.RUN_ALL:
            submissions_config_to_run = submission_config
        case (
            RunOnlySelectedConfigsOption.RUN_ONLY_FIRST
            | RunOnlySelectedConfigsOption.RUN_ONLY_LAST
            | RunOnlySelectedConfigsOption.RUN_SINGLE_RANDOM
        ):
            print(  # noqa: T201 - We want this submission script to print this output
                f"<<< NOTE: {run_only_selected_configs_option = }",
            )
            print(  # noqa: T201 - We want this submission script to print this output
                "<<< NOTE: Running only a selection of the options in the given argument lists.",
            )
            submissions_config_to_run: SubmissionConfig = pick_selected_options_in_each_list(
                submission_config=submission_config,
                run_only_selected_configs_option=run_only_selected_configs_option,
            )
        case _:
            msg: str = f"Unknown {run_only_selected_configs_option = }"
            raise ValueError(
                msg,
            )

    run_task(
        submission_config=submissions_config_to_run,
        task=task,
        run_option=run_option,
    )


def run_task(
    submission_config: SubmissionConfig,
    task: Task,
    *,
    run_option: RunOption = RunOption.DO_SUBMISSION,
) -> None:
    """Run a task with the given configuration."""
    match task:
        case Task.LOCAL_ESTIMATES_COMPUTATION:
            submission_config.python_script_name = "run_local_estimates.py"
            submission_config.relative_python_script_folder = pathlib.Path(
                "topollm",
                "analysis",
                "local_estimates_computation",
            )
        case Task.PIPELINE:
            submission_config.python_script_name = "run_pipeline_compute_embeddings_and_data_prep_and_local_estimate.py"
            submission_config.relative_python_script_folder = pathlib.Path(
                "topollm",
                "pipeline_scripts",
            )
        case Task.PERPLEXITY:
            submission_config.python_script_name = "run_compute_perplexity.py"
            submission_config.relative_python_script_folder = pathlib.Path(
                "topollm",
                "model_inference",
                "perplexity",
            )
        case Task.FINETUNING:
            submission_config.python_script_name = "run_finetune_language_model_on_huggingface_dataset.py"
            submission_config.relative_python_script_folder = pathlib.Path(
                "topollm",
                "model_finetuning",
            )
        case _:
            msg = f"Unknown {task = }"
            raise ValueError(msg)

    command: list[str] = submission_config.get_command(
        task=task,  # type: ignore - StrEnum typing problems
    )
    print(  # noqa: T201 - We want this submission script to print the command
        "Running command:\n",
        " ".join(command),
    )

    if run_option == RunOption.DRY_RUN:
        print(  # noqa: T201 - We want this submission script to print this output
            "@@@@ Dry run, not actually running the command. @@@@",
        )
    else:
        subprocess.run(
            args=command,
            check=True,
        )

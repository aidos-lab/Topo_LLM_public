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

from enum import StrEnum, auto


class RunOption(StrEnum):
    """Options for selecting a dry run."""

    DO_SUBMISSION = auto()
    DRY_RUN = auto()


class ExperimentStage(StrEnum):
    """Options for the experiment stage."""

    COMPUTE_EMBEDDINGS_PLUS_SINGLE_PIPELINE_RUN = auto()
    SKIP_COMPUTE_EMBEDDINGS_BUT_DO_MULTIPLE_PIPELINE_RUNS = auto()


class ExperimentSelector(StrEnum):
    """Options for the experiment selector."""

    # >>> Sensitivity analysis

    # Influence of data subsampling number of samples
    SENSITIVITY_ANALYIS_MULTIWOZ21_DIFFERENT_DATA_SUBSAMPLING_NUMBER_OF_SAMPLES = auto()
    SENSITIVITY_ANALYSIS_REDDIT_DIFFERENT_DATA_SUBSAMPLING_NUMBER_OF_SAMPLES = auto()

    COARSE_CHECKPOINT_RESOLUTION = auto()
    EXPLORATORY_DROPOUT_ANALYSIS_COARSE_CHECKPOINT_RESOLUTION = auto()
    TINY_DROPOUT_VARIATIONS_COARSE_CHECKPOINT_RESOLUTION = auto()
    FIXED_PARAMETERS_HIGH_CHECKPOINT_RESOLUTION = auto()

    REGULAR_TOKEN_EMBEDDINGS = auto()
    MASKED_TOKEN_EMBEDDINGS = auto()

    REGULAR_TOKEN_EMBEDDINGS_MULTIPLE_LAYERS_SINGLE_SAMPLE = auto()
    MASKED_TOKEN_EMBEDDINGS_LAST_LAYER_SINGLE_SAMPLE = auto()

    REGULAR_TOKEN_EMBEDDINGS_MULTIPLE_LOCAL_ESTIMATES_POINTWISE_ABSOLUTE_N_NEIGHBORS = auto()


class CheckpointNoListOption(StrEnum):
    """Options for the checkpoint number list."""

    FULL = auto()
    ONLY_BEGINNING_AND_MIDDLE_AND_END = auto()
    SELECTED = auto()

    # Fixed checkpoints
    FIXED_2800 = auto()


class DataListOption(StrEnum):
    """Options for the data list."""

    DEBUG = auto()
    FULL = auto()
    MANUAL_IN_PYTHON_SCRIPT = auto()
    # Selecting only one dataset
    ICLR_VALIDATION_ONLY = auto()
    MULTIWOZ21_ONLY = auto()
    MULTIWOZ21_VALIDATION_ONLY = auto()
    REDDIT_ONLY = auto()
    REDDIT_VALIDATION_ONLY = auto()
    SGD_VALIDATION_ONLY = auto()
    WIKITEXT_ONLY = auto()
    WIKITEXT_VALIDATION_ONLY = auto()
    # Select certain data splits
    TRAIN_SPLIT_ONLY = auto()
    VALIDATION_SPLIT_ONLY = auto()
    # Mixing two datasets
    MULTIWOZ21_AND_REDDIT = auto()
    MULTIWOZ21_TRAIN_AND_REDDIT_TRAIN = auto()
    MULTIWOZ21_VALIDATION_AND_REDDIT_VALIDATION = auto()
    # Mixing three datasets
    ICLR_VALIDATION_AND_SGD_VALIDATION_AND_WIKITEXT_VALIDATION = auto()


class DataSubsamplingNumberOfSamplesListOption(StrEnum):
    """Options for the data number of samples list."""

    NONE = auto()
    FIXED_3000 = auto()
    FIXED_10000 = auto()
    FIXED_12000 = auto()
    FIXED_16000 = auto()
    FIXED_22000 = auto()
    RANGE_START_2000_STOP_12000_STEP_2000 = auto()
    RANGE_START_2000_STOP_18000_STEP_2000 = auto()  # For exhausting the entire multiwoz21 validation and test sets
    RANGE_START_12000_STOP_18000_STEP_2000 = auto()
    RANGE_START_2000_STOP_24000_STEP_2000 = (
        auto()
    )  # For exhausting the entire one-year-of-tsla-on-reddit validation and test sets
    RANGE_START_12000_STOP_24000_STEP_2000 = auto()


class DataSubsamplingSamplingSeedListOption(StrEnum):
    """Options for the seed lists for the data subsampling."""

    NONE = auto()
    DEFAULT = auto()
    FIXED_777 = auto()
    TWO_SEEDS = auto()
    THREE_SEEDS = auto()
    FIVE_SEEDS = auto()
    TEN_SEEDS = auto()
    TWENTY_SEEDS = auto()


class FinetuningDatasetsListOption(StrEnum):
    """Options for the finetuning dataset list."""

    DEBUG = auto()
    MANUAL_IN_PYTHON_SCRIPT = auto()
    # Single datasets
    ICLR_SMALL = auto()
    MULTIWOZ21_SMALL = auto()
    MULTIWOZ21_FULL = auto()
    REDDIT_SMALL = auto()
    REDDIT_FULL = auto()
    SGD_SMALL = auto()
    WIKITEXT_SMALL = auto()
    # Multiple datasets in separate runs
    MULTIWOZ21_AND_REDDIT_FULL = auto()
    MULTIWOZ21_AND_REDDIT_SMALL = auto()


class ModelGroupOption(StrEnum):
    """Options for specifying the model and finetuning regime."""

    ROBERTA_BASE_WITHOUT_MODIFICATIONS = auto()
    ROBERTA_BASE_FINETUNED_FOR_FEW_EPOCHS_OLD_AND_NEW_DATA_SINGLE_SEED_LAST_CHECKPOINT = auto()
    ROBERTA_BASE_FINETUNED_FOR_MANY_EPOCHS = auto()


class FinetuningRegimeOption(StrEnum):
    """Options for the finetuning regime."""

    FEW_EPOCHS = auto()
    MANY_EPOCHS_WITH_OVERFITTING_RISK = auto()


class LanguageModelListOption(StrEnum):
    """Options for the language model list."""

    ONLY_ROBERTA_BASE = auto()

    # Models fine-tuned on "old data" (i.e., data which was part of the pretraining data)
    # and models fine-tuned on "new data" (i.e., data which was created after the model was pre-trained)
    FINETUNED_ON_OLD_AND_NEW_DATA_FEW_EPOCHS_FROM_ROBERTA_BASE = auto()

    # Selected models
    SELECTED_FINETUNED_FEW_EPOCHS_FROM_ROBERTA_BASE = auto()
    SELECTED_FINETUNED_MANY_EPOCHS_FROM_ROBERTA_BASE = auto()

    FULL_FINETUNED_FEW_EPOCHS_FROM_ROBERTA_BASE = auto()

    # Models finetuned with different dropout rates
    WITH_005_015_02_DROPOUT_FINETUNED_ON_MULTIWOZ_SMALL_MANY_EPOCHS_FROM_ROBERTA_BASE = auto()
    WITH_006_007_DROPOUT_FINETUNED_ON_MULTIWOZ_SMALL_MANY_EPOCHS_FROM_ROBERTA_BASE = auto()

    SETSUMBT_SELECTED = auto()


class SeedListOption(StrEnum):
    """Options for the seed lists."""

    DO_NOT_SET = auto()
    ONE_SEED = auto()
    TWO_SEEDS = auto()
    FIVE_SEEDS = auto()
    FIXED_SEED_1234 = auto()
    FIXED_SEEDS_1234_1235_1236 = auto()
    FIXED_SEEDS_1235_1236 = auto()


class EmbeddingsDataPrepSamplingSeedListOption(StrEnum):
    """Options for the seed lists for the embeddings data preparation sampling."""

    DEFAULT = auto()
    TWO_SEEDS = auto()
    FIVE_SEEDS = auto()
    TEN_SEEDS = auto()
    TWENTY_SEEDS = auto()


class EmbeddingsDataPrepNumSamplesListOption(StrEnum):
    """Options for the number of samples in the embeddings data preparation sampling."""

    DEFAULT = auto()
    SINGLE_CHOICE_50000 = auto()
    SINGLE_CHOICE_100000 = auto()
    SINGLE_CHOICE_150000 = auto()
    SINGLE_CHOICE_250000 = auto()
    FIVE_CHOICES_10000_STEPS = auto()


class LocalEstimatesFilteringNumSamplesListOption(StrEnum):
    """Options for the number of samples for local estimates filtering."""

    DEFAULT = auto()
    SINGLE_CHOICE_60000 = auto()
    FEW_SMALL_STEPS_NUM_SAMPLES = auto()
    MEDIUM_SMALL_STEPS_NUM_SAMPLES = auto()
    UP_TO_30000_WITH_STEP_SIZE_2500_NUM_SAMPLES = auto()
    UP_TO_30000_WITH_STEP_SIZE_5000_NUM_SAMPLES = auto()
    UP_TO_50000_WITH_STEP_SIZE_5000_NUM_SAMPLES = auto()
    UP_TO_90000_WITH_STEP_SIZE_5000_NUM_SAMPLES = auto()
    UP_TO_90000_WITH_STEP_SIZE_10000_NUM_SAMPLES = auto()
    UP_TO_100000_WITH_STEP_SIZE_20000_NUM_SAMPLES = auto()


class LocalEstimatesPointwiseAbsoluteNNeighborsListOption(StrEnum):
    """Options for the number of neighbors for pointwise absolute local estimates."""

    DEFAULT = auto()
    SINGLE_CHOICE_128 = auto()
    POWERS_OF_TWO_UP_TO_1024 = auto()


class RunOnlySelectedConfigsOption(StrEnum):
    """Options to run only a single or selected config."""

    RUN_ALL = auto()
    RUN_ONLY_FIRST = auto()
    RUN_ONLY_LAST = auto()
    RUN_SINGLE_RANDOM = auto()

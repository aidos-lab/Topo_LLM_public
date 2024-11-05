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
from pickle import NONE


class CheckpointNoListOption(StrEnum):
    """Options for the checkpoint number list."""

    FULL = auto()
    ONLY_BEGINNING_AND_MIDDLE_AND_END = auto()
    SELECTED = auto()


class DataListOption(StrEnum):
    """Options for the data list."""

    DEBUG = auto()
    FULL = auto()
    MANUAL_IN_PYTHON_SCRIPT = auto()
    TRAIN_ONLY = auto()
    MULTIWOZ21_AND_REDDIT = auto()
    MULTIWOZ21_TRAIN_AND_REDDIT_TRAIN = auto()
    MULTIWOZ21_ONLY = auto()
    REDDIT_ONLY = auto()


class DataNumberOfSamplesListOption(StrEnum):
    """Options for the data number of samples list."""

    NONE = auto()
    FIXED_3000 = auto()


class FinetuningDatasetsListOption(StrEnum):
    """Options for the finetuning dataset list."""

    DEBUG = auto()
    MANUAL_IN_PYTHON_SCRIPT = auto()


class FinetuningRegimeOption(StrEnum):
    """Options for the finetuning regime."""

    FEW_EPOCHS = auto()
    MANY_EPOCHS_WITH_OVERFITTING_RISK = auto()


class LanguageModelListOption(StrEnum):
    """Options for the language model list."""

    ONLY_ROBERTA_BASE = auto()
    SELECTED_FINETUNED_FEW_EPOCHS_FROM_ROBERTA_BASE = auto()
    SELECTED_FINETUNED_MANY_EPOCHS_FROM_ROBERTA_BASE = auto()
    FULL_FINETUNED_FEW_EPOCHS_FROM_ROBERTA_BASE = auto()
    SETSUMBT_SELECTED = auto()


class SeedListOption(StrEnum):
    """Options for the seed lists."""

    DO_NOT_SET = auto()
    ONE_SEED = auto()
    TWO_SEEDS = auto()
    FIVE_SEEDS = auto()


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
    FIVE_CHOICES_10000_STEPS = auto()


class LocalEstimatesFilteringNumSamplesListOption(StrEnum):
    """Options for the number of samples for local estimates filtering."""

    DEFAULT = auto()
    FEW_SMALL_STEPS_NUM_SAMPLES = auto()
    MEDIUM_SMALL_STEPS_NUM_SAMPLES = auto()
    UP_TO_30000_WITH_STEP_SIZE_2500_NUM_SAMPLES = auto()
    UP_TO_30000_WITH_STEP_SIZE_5000_NUM_SAMPLES = auto()
    UP_TO_50000_WITH_STEP_SIZE_5000_NUM_SAMPLES = auto()
    UP_TO_90000_WITH_STEP_SIZE_5000_NUM_SAMPLES = auto()
    UP_TO_90000_WITH_STEP_SIZE_10000_NUM_SAMPLES = auto()


class LocalEstimatesPointwiseAbsoluteNNeighborsListOption(StrEnum):
    """Options for the number of neighbors for pointwise absolute local estimates."""

    DEFAULT = auto()
    SINGLE_CHOICE_128 = auto()
    POWERS_OF_TWO_UP_TO_1024 = auto()

# coding=utf-8
#
# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (ruppik@hhu.de)
#
# This code was generated with the help of AI writing assistants
# including GitHub Copilot, ChatGPT, Bing Chat.
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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Imports

# System imports
from enum import Enum, unique

# Third-party imports

try:
    # Try to import StrEnum from the standard library (Python 3.11 and later)
    from enum import StrEnum  # type: ignore
except ImportError:
    # Fallback to the strenum package for Python 3.10 and below.
    # Run `python3 -m pip install strenum` for python < 3.11
    from strenum import StrEnum

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# ==============================


@unique
class Level(StrEnum):
    TOKEN = "token"
    WVFS = "wvfs"
    DATASET_ENTRY = "dataset_entry"


@unique
class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    FULL = "full"


# # # #
# dicts for translating between short and long names
level_short_to_long: dict[str, str] = {
    "token": "token",
    "wvfs": "word-via-first-subtoken",
    "wvls": "word-via-last-subtoken",
}

masking_mode_short_to_long: dict[str, str] = {
    "hsout": "hidden-state-of-unmasked-token",
}


# ==============================


@unique
class DataLoaderKey(StrEnum):
    TRAINING = "train"
    VALIDATION = "validation"
    TESTING = "test"
    PREDICTION = "prediction"


@unique
class ExperimentMode(StrEnum):
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"
    PREDICTION = "prediction"


def experiment_mode_to_data_loader_key(
    experiment_mode: ExperimentMode,
) -> DataLoaderKey:
    if experiment_mode == ExperimentMode.TRAINING:
        return DataLoaderKey.TRAINING
    elif experiment_mode == ExperimentMode.VALIDATION:
        return DataLoaderKey.VALIDATION
    elif experiment_mode == ExperimentMode.TESTING:
        return DataLoaderKey.TESTING
    elif experiment_mode == ExperimentMode.PREDICTION:
        return DataLoaderKey.PREDICTION
    else:
        raise ValueError(f"Unknown experiment_mode: {experiment_mode}")


# ==============================


@unique
class DialogueDatasetDesc(StrEnum):
    MULTIWOZ21 = "multiwoz21"
    MULTIWOZ21_TRAIN = "multiwoz21-train"
    MULTIWOZ21_VALIDATION = "multiwoz21-validation"
    MULTIWOZ21_TEST = "multiwoz21-test"
    SGD = "sgd"
    SGD_TRAIN = "sgd-train"
    SGD_VALIDATION = "sgd-validation"
    SGD_TEST = "sgd-test"


@unique
class SplitMode(StrEnum):
    FULL = "full"
    COLUMN = "column"
    RANDOM = "random"


@unique
class SubtokenExtractionMethod(StrEnum):
    FIRST_SUBTOKEN = "first_subtoken"
    LAST_SUBTOKEN = "last_subtoken"


translate_subtoken_extraction_to_level: dict[
    SubtokenExtractionMethod, dict[str, str]
] = {
    SubtokenExtractionMethod.FIRST_SUBTOKEN: {
        "value_short": "wvfs",
        "value_long": level_short_to_long["wvfs"],
    },
    SubtokenExtractionMethod.LAST_SUBTOKEN: {
        "value_short": "wvls",
        "value_long": level_short_to_long["wvls"],
    },
}


@unique
class DatasetMode(StrEnum):
    DATASETS_CONTEXTUAL_DIALOGUES = "datasets_contextual_dialogues"
    DATASETS_NER_EXPERIMENTS = "datasets_ner_experiments"


@unique
class WordSeparationMethod(StrEnum):
    # Option 1:
    # word_separation_method = "from_tokenizer"
    # means: use the word separation created by the tokenizer
    # via the tokenizer.encode() method and returned word_ids.
    # Note that in that case, you have no control over the word separation.
    #
    # Option 2:
    # word_separation_method = "from_dataset"
    # means: use the word separation created by the dataset
    # (for instance, the CoNLL2003 dataset comes separated into words,
    # with each word associated with a label on a separate line)
    FROM_TOKENIZER = "from_tokenizer"
    FROM_DATASET = "from_dataset"


@unique
class DataSourceMode(StrEnum):
    HUGGINGFACE_DATASET = "huggingface_dataset"
    CONVLAB_UNIFIED_DATAFORMAT = "convlab_unified_dataformat"


# ==============================
# Enums used for neighborhoods configurations
# ==============================


@unique
class NeighborhoodsCenterMethod(StrEnum):
    IGNORE = "ignore"
    ADD_CENTER = "add_center"
    ADDIFNOT = "addifnot"


@unique
class PrecomputedNeighborsMetric(StrEnum):
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"


@unique
class RipserMetric(StrEnum):
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"


@unique
class NeighborhoodsKeyDesc(StrEnum):
    MULTIWOZ21_TRAIN = "multiwoz21-train"
    SGD_TRAIN = "sgd-train"
    MULTIWOZ21_TRAIN_AND_SGD_TRAIN = "multiwoz21-train-and-sgd-train"


@unique
class WhichEmbeddings(StrEnum):
    KEY = "key"
    QUERY = "query"


@unique
class DistancesOrIndices(StrEnum):
    DISTANCES = "distances"
    INDICES = "indices"


# ==============================
# Enums used for configuration parameters
# of the BioTagger PyTorch models
# ==============================


@unique
class FeatureType(StrEnum):
    LM = "lm"
    LM_C_PIS_H0 = "lm_C_PIs_H0"
    LM_C_WASSERSTEIN_H0_H1 = "lm_C_wasserstein_H0_H1"
    LM_C_CODENSITY = "lm_C_codensity"
    LM_C_LOCALDIM = "lm_C_localdim"
    LM_C_PERPLEXITY = "lm_C_perplexity"


@unique
class AdditionalLayerNormForInProjection(StrEnum):
    NONE = "None"
    BEFORE = "before"
    AFTER = "after"


@unique
class EncodedSliceCombinationMode(StrEnum):
    ADD = "add"
    CONCATENATE = "concatenate"
    NONE = "none"


@unique
class InProjectionType(StrEnum):
    LINEAR = "linear"
    MLP = "mlp"
    SLICED_MLP = "sliced_mlp"


@unique
class ModelSelectionMetric(StrEnum):
    F1_MACRO = "f1_macro"
    PHRASAL_OVERALL_F1 = "phrasal_overall_f1"
    TRAINING_LOSS = "training_loss"


@unique
class SliceNormType(StrEnum):
    NONE = "None"
    INDIVIDUAL = "individual"
    SHARED = "shared"


@unique
class TaggerType(StrEnum):
    ROBERTA = "roberta"


@unique
class LrSchedulerType(StrEnum):
    CONSTANT = "constant"
    LINEAR_WITH_WARMUP = "linear_with_warmup"

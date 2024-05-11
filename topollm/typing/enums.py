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

"""Enums used in the configuration classes."""

from enum import Enum, IntEnum, unique

from strenum import StrEnum

try:
    # Try to import StrEnum from the standard library (Python 3.11 and later)
    from enum import StrEnum  # type: ignore - loading of the module is not guaranteed
except ImportError:
    # Fallback to the strenum package for Python 3.10 and below.
    # Run `python3 -m pip install strenum` for python < 3.11
    from strenum import StrEnum  # type: ignore - loading of the module is not guaranteed

if not issubclass(
    StrEnum,
    Enum,
):
    msg = "StrEnum should be a subclass of Enum"
    raise TypeError(msg)


# ==============================


class Verbosity(IntEnum):
    """Verbosity level."""

    QUIET = 0
    NORMAL = 1
    VERBOSE = 2
    DEBUG = 3


@unique
class PreferredTorchBackend(StrEnum):  # type: ignore
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"


# ==============================


@unique
class DatasetType(StrEnum):  # type: ignore
    HUGGINGFACE_DATASET = "huggingface_dataset"


# ==============================


@unique
class ArrayStorageType(StrEnum):  # type: ignore
    ZARR = "zarr"


@unique
class MetadataStorageType(StrEnum):  # type: ignore
    XARRAY = "xarray"
    PICKLE = "pickle"


# ==============================


@unique
class AggregationType(StrEnum):  # type: ignore
    CONCATENATE = "concatenate"
    MEAN = "mean"


@unique
class Level(StrEnum):  # type: ignore
    TOKEN = "token"
    WVFS = "wvfs"
    DATASET_ENTRY = "dataset_entry"


@unique
class Split(StrEnum):  # type: ignore
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
class DataLoaderKey(StrEnum):  # type: ignore
    TRAINING = "train"
    VALIDATION = "validation"
    TESTING = "test"
    PREDICTION = "prediction"


@unique
class ExperimentMode(StrEnum):  # type: ignore
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"
    PREDICTION = "prediction"


# ==============================


@unique
class DialogueDatasetDesc(StrEnum):  # type: ignore
    MULTIWOZ21 = "multiwoz21"
    MULTIWOZ21_TRAIN = "multiwoz21-train"
    MULTIWOZ21_VALIDATION = "multiwoz21-validation"
    MULTIWOZ21_TEST = "multiwoz21-test"
    SGD = "sgd"
    SGD_TRAIN = "sgd-train"
    SGD_VALIDATION = "sgd-validation"
    SGD_TEST = "sgd-test"


@unique
class DataSplitMode(StrEnum):  # type: ignore
    DO_NOTHING = "do_nothing"
    PROPORTIONS = "proportions"


@unique
class SubtokenExtractionMethod(StrEnum):  # type: ignore
    FIRST_SUBTOKEN = "first_subtoken"
    LAST_SUBTOKEN = "last_subtoken"


@unique
class DatasetMode(StrEnum):  # type: ignore
    DATASETS_CONTEXTUAL_DIALOGUES = "datasets_contextual_dialogues"
    DATASETS_NER_EXPERIMENTS = "datasets_ner_experiments"


@unique
class WordSeparationMethod(StrEnum):  # type: ignore
    # Option 1:
    # WordSeparationMethod.FROM_TOKENIZER means:
    # Use the word separation created by the tokenizer
    # via the tokenizer.encode() method and returned word_ids.
    # Note that in that case, you have no control over the word separation.
    FROM_TOKENIZER = "from_tokenizer"
    # Option 2:
    # WordSeparationMethod.FROM_DATASET means:
    # Use the word separation created by the dataset
    # (for instance, the CoNLL2003 dataset comes separated into words,
    # with each word associated with a label on a separate line)
    FROM_DATASET = "from_dataset"


# ==============================
# Enums used for neighborhoods configurations
# ==============================


@unique
class NeighborhoodsCenterMethod(StrEnum):  # type: ignore
    IGNORE = "ignore"
    ADD_CENTER = "add_center"
    ADDIFNOT = "addifnot"


@unique
class PrecomputedNeighborsMetric(StrEnum):  # type: ignore
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"


@unique
class RipserMetric(StrEnum):  # type: ignore
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"


@unique
class NeighborhoodsKeyDesc(StrEnum):  # type: ignore
    MULTIWOZ21_TRAIN = "multiwoz21-train"
    SGD_TRAIN = "sgd-train"
    MULTIWOZ21_TRAIN_AND_SGD_TRAIN = "multiwoz21-train-and-sgd-train"


@unique
class WhichEmbeddings(StrEnum):  # type: ignore
    KEY = "key"
    QUERY = "query"


@unique
class DistancesOrIndices(StrEnum):  # type: ignore
    DISTANCES = "distances"
    INDICES = "indices"


# ==============================
# Enums used for configuration parameters
# of the BioTagger PyTorch models
# ==============================


@unique
class FeatureType(StrEnum):  # type: ignore
    LM = "lm"
    LM_C_PIS_H0 = "lm_C_PIs_H0"
    LM_C_WASSERSTEIN_H0_H1 = "lm_C_wasserstein_H0_H1"
    LM_C_CODENSITY = "lm_C_codensity"
    LM_C_LOCALDIM = "lm_C_localdim"
    LM_C_PERPLEXITY = "lm_C_perplexity"


@unique
class AdditionalLayerNormForInProjection(StrEnum):  # type: ignore
    NONE = "None"
    BEFORE = "before"
    AFTER = "after"


@unique
class EncodedSliceCombinationMode(StrEnum):  # type: ignore
    ADD = "add"
    CONCATENATE = "concatenate"
    NONE = "none"


@unique
class InProjectionType(StrEnum):  # type: ignore
    LINEAR = "linear"
    MLP = "mlp"
    SLICED_MLP = "sliced_mlp"


@unique
class ModelSelectionMetric(StrEnum):  # type: ignore
    F1_MACRO = "f1_macro"
    PHRASAL_OVERALL_F1 = "phrasal_overall_f1"
    TRAINING_LOSS = "training_loss"


@unique
class SliceNormType(StrEnum):  # type: ignore
    NONE = "None"
    INDIVIDUAL = "individual"
    SHARED = "shared"


@unique
class TaggerType(StrEnum):  # type: ignore
    ROBERTA = "roberta"


@unique
class LrSchedulerType(StrEnum):  # type: ignore
    CONSTANT = "constant"
    LINEAR_WITH_WARMUP = "linear_with_warmup"


# ==============================
# Enums used for finetuning parameters
# ==============================


@unique
class FinetuningMode(StrEnum):
    """The different modes for finetuning."""

    STANDARD = "standard"
    LORA = "lora"


@unique
class LMmode(StrEnum):
    """The different types of language models."""

    MLM = "MLM"  # masked language model
    CLM = "CLM"  # causal language model


@unique
class TokenizerModifierMode(StrEnum):
    """Modes for modifying the tokenizer."""

    DO_NOTHING = "do_nothing"
    ADD_PADDING_TOKEN = "add_padding_token"  # noqa: S105 - not a password token


@unique
class GradientModifierMode(StrEnum):
    """Modes for modifying the gradients during finetuning."""

    DO_NOTHING = "do_nothing"
    FREEZE_LAYERS = "freeze_layers"


# ==============================
# Enums used for perplexity
# ==============================


@unique
class MLMPseudoperplexityGranularity(StrEnum):
    """The different modes for computing the pseudoperplexity of a masked language model."""

    TOKEN = "TOKEN"  # noqa: S105 - this is not a password
    SENTENCE = "SENTENCE"


@unique
class PerplexityContainerSaveFormat(StrEnum):
    """The different formats in which the perplexity container can be saved."""

    PICKLE = "pickle"
    JSONL = "jsonl"

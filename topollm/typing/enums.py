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

from enum import Enum, IntEnum, auto, unique

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
class PreferredTorchBackend(StrEnum):
    """The preferred backend for PyTorch."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"


# ==============================


@unique
class JobRunMode(StrEnum):
    """The different modes for running jobs."""

    LOCAL = "local"
    HHU_HILBERT = "hhu_hilbert"


# ==============================


@unique
class DatasetType(StrEnum):
    """The different types of datasets."""

    HUGGINGFACE_DATASET = "huggingface_dataset"


# ==============================


@unique
class ArrayStorageType(StrEnum):
    """The different types of array storage."""

    ZARR = "zarr"


@unique
class MetadataStorageType(StrEnum):
    """The different types of metadata storage."""

    XARRAY = "xarray"
    PICKLE = "pickle"


# ==============================


@unique
class AggregationType(StrEnum):
    """The different types of aggregation for the embedding vectors."""

    CONCATENATE = "concatenate"
    MEAN = "mean"


@unique
class Level(StrEnum):
    """The different levels for the embedding vector extraction."""

    TOKEN = "token"  # noqa: S105 - not a password token
    WVFS = "wvfs"
    DATASET_ENTRY = "dataset_entry"


@unique
class Split(StrEnum):
    """Splits of a dataset."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    FULL = "full"


@unique
class MaskingMode(StrEnum):
    """The different masking modes."""

    NO_MASKING = auto()


# ==============================


@unique
class DescriptionType(StrEnum):
    """The different types of descriptions."""

    SHORT = "short"
    LONG = "long"


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
class ZeroVectorHandlingMode(StrEnum):
    """The different modes for handling zero vectors."""

    KEEP = "keep"
    REMOVE = "remove"


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
class DataSplitMode(StrEnum):
    """The different modes for splitting the data."""

    DO_NOTHING = "do_nothing"
    PROPORTIONS = "proportions"


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


@unique
class LrSchedulerType(StrEnum):
    """The different types of learning rate schedulers."""

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


class TaskType(StrEnum):
    """The different types of tasks."""

    SEQUENCE_CLASSIFICATION = auto()
    TOKEN_CLASSIFICATION = auto()
    MASKED_LM = auto()
    CAUSAL_LM = auto()


@unique
class TokenizerModifierMode(StrEnum):
    """Modes for modifying the tokenizer."""

    DO_NOTHING = auto()
    ADD_PADDING_TOKEN = auto()


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

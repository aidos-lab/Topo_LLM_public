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

    CPU = auto()
    CUDA = auto()
    MPS = auto()
    AUTO = auto()


# ==============================


@unique
class JobRunMode(StrEnum):
    """The different modes for running jobs."""

    LOCAL = auto()
    HHU_HILBERT = auto()


# ==============================


@unique
class DatasetType(StrEnum):
    """The different types of datasets."""

    HUGGINGFACE_DATASET = auto()
    HUGGINGFACE_DATASET_NAMED_ENTITY = auto()


# ==============================


@unique
class ArrayStorageType(StrEnum):
    """The different types of array storage."""

    ZARR = auto()


@unique
class MetadataStorageType(StrEnum):
    """The different types of metadata storage."""

    XARRAY = auto()
    PICKLE = auto()


# ==============================


@unique
class AggregationType(StrEnum):
    """The different types of aggregation for the embedding vectors."""

    CONCATENATE = auto()
    MEAN = auto()


@unique
class Level(StrEnum):
    """The different levels for the embedding vector extraction."""

    TOKEN = auto()  # noqa: S105 - not a password token
    WVFS = auto()
    DATASET_ENTRY = auto()


@unique
class Split(StrEnum):
    """Splits of a dataset."""

    TRAIN = auto()
    VALIDATION = auto()
    TEST = auto()
    FULL = auto()


@unique
class MaskingMode(StrEnum):
    """The different masking modes."""

    NO_MASKING = auto()


# ==============================


@unique
class DescriptionType(StrEnum):
    """The different types of descriptions."""

    SHORT = auto()
    LONG = auto()


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

    KEEP = auto()
    REMOVE = auto()


# ==============================


@unique
class DataSplitMode(StrEnum):
    """The different modes for splitting the data."""

    DO_NOTHING = auto()
    PROPORTIONS = auto()


@unique
class WordSeparationMethod(StrEnum):
    """The different methods for separating words."""

    # Option 1:
    # WordSeparationMethod.FROM_TOKENIZER means:
    # Use the word separation created by the tokenizer
    # via the tokenizer.encode() method and returned word_ids.
    # Note that in that case, you have no control over the word separation.
    FROM_TOKENIZER = auto()
    # Option 2:
    # WordSeparationMethod.FROM_DATASET means:
    # Use the word separation created by the dataset
    # (for instance, the CoNLL2003 dataset comes separated into words,
    # with each word associated with a label on a separate line)
    FROM_DATASET = auto()


@unique
class LrSchedulerType(StrEnum):
    """The different types of learning rate schedulers."""

    CONSTANT = auto()
    LINEAR = auto()


# ==============================
# Enums used for embeddings data preparation
# ==============================


@unique
class EmbeddingsDataPrepSamplingMode(StrEnum):
    """The different modes for sampling in the embeddings data prep step."""

    RANDOM = auto()
    TAKE_FIRST = auto()


# ==============================
# Enums used for finetuning parameters
# ==============================


@unique
class FinetuningMode(StrEnum):
    """The different modes for finetuning."""

    STANDARD = auto()
    LORA = auto()


class ComputeMetricsMode(StrEnum):
    """Compute metrics to use in the finetuning Trainer."""

    NONE = auto()
    FROM_TASK_TYPE = auto()


@unique
class LMmode(StrEnum):
    """The different types of language models."""

    MLM = auto()  # masked language model
    CLM = auto()  # causal language model


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

    DO_NOTHING = auto()
    FREEZE_LAYERS = auto()


@unique
class TrainerModifierMode(StrEnum):
    """Modes for modifying the trainer during finetuning."""

    DO_NOTHING = auto()
    ADD_WANDB_PREDICTION_PROGRESS_CALLBACK = auto()


# ==============================
# Enums used for perplexity
# ==============================


@unique
class MLMPseudoperplexityGranularity(StrEnum):
    """The different modes for computing the pseudoperplexity of a masked language model."""

    TOKEN = auto()
    SENTENCE = auto()


@unique
class PerplexityContainerSaveFormat(StrEnum):
    """The different formats in which the perplexity container can be saved."""

    LIST_AS_JSONL = auto()
    LIST_AS_PICKLE = auto()
    CONCATENATED_DATAFRAME_AS_CSV = auto()
    CONCATENATED_ARRAY_AS_ZARR = auto()


class SubmissionMode(StrEnum):
    """Submission mode for running scripts."""

    LOCAL = auto()
    HPC_SUBMISSION = auto()


class Task(StrEnum):
    """Enumeration of tasks."""

    LOCAL_ESTIMATES_COMPUTATION = auto()
    PIPELINE = auto()
    PERPLEXITY = auto()
    FINETUNING = auto()

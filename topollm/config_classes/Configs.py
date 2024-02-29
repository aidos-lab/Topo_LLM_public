# coding=utf-8
#
# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Julius von Rohrscheidt (julius.rohrscheidt@helmholtz-muenchen.de)
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

# Standard library imports
import json
from os import PathLike
import pathlib
import pprint
from abc import ABC, abstractmethod
from typing import IO

# Third party imports
from pydantic import BaseModel, Field

# Local imports
from topollm.config_classes.ConfigBaseModel import ConfigBaseModel
from topollm.config_classes.enums import (
    AggregationType,
    Level,
    Split,
    DatasetType,
    StorageType,
)

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Globals

# END Globals
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class DataConfig(ConfigBaseModel):
    """
    Configurations for specifying data.
    """

    column_name: str = Field(
        ...,
        title="Column name to use for computing embeddings.",
        description="The column name to use for computing embeddings.",
    )

    context: str = Field(
        ...,
        title="Context to use for computing embeddings.",
        description="The context to use for computing embeddings.",
    )

    dataset_identifier: str = Field(
        ...,
        title="Dataset identifier for huggingface datasets.",
        description="The dataset identifier for the huggingface datasets "
        "to use for computing embeddings.",
    )

    dataset_type: DatasetType = Field(
        ...,
        title="Dataset type.",
        description="The dataset type.",
    )

    number_of_samples: int = Field(
        ...,
        title="Number of samples to use for computing embeddings.",
        description="The number of samples to use for computing embeddings.",
    )

    split: Split = Field(
        ...,
        title="Split to use for computing embeddings.",
        description="The split to use for computing embeddings.",
    )


class DatasetMapConfig(ConfigBaseModel):
    """
    Configurations for specifying dataset map.
    """

    batch_size: int = Field(
        ...,
        title="Batch size for mapping tokenization on dataset.",
        description="The batch size for mapping tokenization on dataset.",
    )

    num_proc: int = Field(
        ...,
        title="Number of processes for mapping tokenization on dataset.",
        description="The number of processes for mapping tokenization on dataset.",
    )


class StorageConfig(ConfigBaseModel):
    """
    Configurations for specifying storage.
    """

    storage_type: StorageType = Field(
        ...,
        title="Storage type.",
        description="The storage type.",
    )


class EmbeddingExtractionConfig(BaseModel):
    layer_indices: list[int]
    aggregation: AggregationType = AggregationType.MEAN


class EmbeddingsConfig(ConfigBaseModel):
    """Configurations for specifying embeddings."""

    dataset_map: DatasetMapConfig = Field(
        ...,
        title="Dataset map configuration.",
        description="The configuration for specifying dataset map.",
    )

    batch_size: int = Field(
        ...,
        title="Batch size for computing embeddings.",
        description="The batch size for computing embeddings.",
    )

    embedding_extraction: EmbeddingExtractionConfig = Field(
        ...,
        title="Embedding extraction configuration.",
        description="The configuration for specifying embedding extraction.",
    )

    huggingface_model_name: str = Field(
        ...,
        title="Model identifier for huggingface transformers model.",
        description="The model identifier for the huggingface transformers model "
        "to use for computing embeddings.",
    )

    level: Level = Field(
        default=Level.TOKEN,
        title="Level to use for computing embeddings.",
        description="The level to use for computing embeddings.",
    )

    max_length: int = Field(
        ...,
        title="Maximum length of the input sequence.",
        description="The maximum length of the input sequence.",
    )

    num_workers: int = Field(
        ...,
        title="Number of workers for dataloader.",
        description="The number of workers for dataloader.",
    )


class PathsConfig(ConfigBaseModel):
    """Configurations for specifying paths."""

    data_dir: pathlib.Path = Field(
        ...,
        title="Data path.",
        description="The path to the data.",
    )

    repository_base_path: pathlib.Path = Field(
        ...,
        title="Repository base path.",
        description="The base path to the repository.",
    )


class MainConfig(ConfigBaseModel):
    """
    Master configuration for computing embeddings.
    """

    data: DataConfig = Field(
        ...,
        title="Data configuration.",
        description="The configuration for specifying data.",
    )

    embeddings: EmbeddingsConfig = Field(
        ...,
        title="Embeddings configuration.",
        description="The configuration for specifying embeddings.",
    )

    paths: PathsConfig = Field(
        ...,
        title="Paths configuration.",
        description="The configuration for specifying paths.",
    )

    storage: StorageConfig = Field(
        ...,
        title="Storage configuration.",
        description="The configuration for specifying storage.",
    )

    verbosity: int = Field(
        default=1,
        title="Verbosity level.",
        description="The verbosity level.",
    )

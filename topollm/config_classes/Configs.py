# coding=utf-8
#
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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Imports

# Standard library imports
from os import PathLike
import pathlib

# Third party imports
from pydantic import BaseModel, Field

# Local imports
from topollm.config_classes.ConfigBaseModel import ConfigBaseModel
from topollm.config_classes.EmbeddingsConfig import EmbeddingsConfig
from topollm.config_classes.enums import (
    AggregationType,
    PreferredTorchBackend,
    Split,
    DatasetType,
    ArrayStorageType,
    MetadataStorageType,
)
from topollm.config_classes.constants import NAME_PREFIXES

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

    dataset_description_string: str = Field(
        ...,
        title="Dataset description string.",
        description=f"The dataset description string. "
        f"This will be used for creating the file paths",
    )

    data_dir: pathlib.Path | None = Field(
        None,
        title="data_dir argument will be passed to huggingface datasets.",
    )

    dataset_path: str = Field(
        ...,
        title="Dataset identifier for huggingface datasets.",
        description="The dataset identifier for the huggingface datasets "
        "to use for computing embeddings.",
    )

    dataset_name: str | None = Field(
        None,
        title="Dataset name.",
        description="The dataset name.",
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

    @property
    def data_config_description(
        self,
    ) -> str:
        return (
            f"{NAME_PREFIXES['data']}{self.dataset_description_string}"
            f"_"
            f"{NAME_PREFIXES['split']}{self.split}"
            f"_"
            f"{NAME_PREFIXES['context']}{self.context}"
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

    array_storage_type: ArrayStorageType = Field(
        ...,
        title="Array storage type.",
        description="The storage type for arrays.",
    )

    metadata_storage_type: MetadataStorageType = Field(
        ...,
        title="Metadata storage type.",
        description="The storage type for metadata.",
    )

    chunk_size: int = Field(
        ...,
        title="Chunk size for storage.",
        description="The chunk size for storage.",
    )


class EmbeddingExtractionConfig(BaseModel):
    layer_indices: list[int]
    aggregation: AggregationType = AggregationType.MEAN  # type: ignore

    @property
    def embedding_extraction_config_description(
        self,
    ) -> str:
        """
        Get the description of the embedding extraction.

        Returns:
            str: The description of the embedding extraction.
        """
        return (
            f"{NAME_PREFIXES['layer']}{str(self.layer_indices)}"
            f"_"
            f"{NAME_PREFIXES['aggregation']}{str(self.aggregation)}"
        )


class LanguageModelConfig(ConfigBaseModel):
    pretrained_model_name_or_path: str | pathlib.Path = Field(
        ...,
        title="Model identifier for huggingface transformers model.",
        description=f"The model identifier for the huggingface transformers model "
        f"to use for computing embeddings.",
    )

    short_model_name: str = Field(
        ...,
        title="Short model name.",
        description="The short model name.",
    )

    masking_mode: str = Field(
        ...,
        title="Masking mode.",
        description="The masking mode.",
    )

    @property
    def lanugage_model_config_description(
        self,
    ) -> str:
        # Construct and return the model parameters description

        return (
            f"{NAME_PREFIXES['model']}{self.short_model_name}"
            f"_"
            f"{NAME_PREFIXES['masking_mode']}{self.masking_mode}"
        )


class TokenizerConfig(ConfigBaseModel):
    add_prefix_space: bool = Field(
        ...,
        title="Add prefix space.",
        description="Whether to add prefix space.",
    )

    max_length: int = Field(
        ...,
        title="Maximum length of the input sequence.",
        description="The maximum length of the input sequence.",
    )

    @property
    def tokenizer_config_description(
        self,
    ) -> str:
        """
        Get the description of the tokenizer config.

        Returns:
            str: The description of the tokenizer.
        """
        return (
            f"{NAME_PREFIXES['add_prefix_space']}{str(self.add_prefix_space)}"
            f"_"
            f"{NAME_PREFIXES['max_length']}{str(self.max_length)}"
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


class TransformationsConfig(ConfigBaseModel):
    normalization: str = Field(
        ...,
        title="Normalization method.",
        description="The normalization method.",
    )

    @property
    def transformation_config_description(
        self,
    ) -> str:
        desc = f"{NAME_PREFIXES['normalization']}"
        desc += f"{self.normalization}"

        return desc


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

    preferred_torch_backend: PreferredTorchBackend = Field(
        ...,
        title="Preferred torch backend.",
        description="The preferred torch backend.",
    )

    storage: StorageConfig = Field(
        ...,
        title="Storage configuration.",
        description="The configuration for specifying storage.",
    )

    transformations: TransformationsConfig = Field(
        ...,
        title="Transformations configuration.",
        description="The configuration for specifying transformations.",
    )

    verbosity: int = Field(
        default=1,
        title="Verbosity level.",
        description="The verbosity level.",
    )

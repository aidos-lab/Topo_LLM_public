# Copyright 2024-2025
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_1 (author1@example.com)
# AUTHOR_2 (author2@example.com)
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

"""Main configuration for all scripts."""

from pydantic import Field

from topollm.config_classes.analysis.analysis_config import AnalysisConfig
from topollm.config_classes.config_base_model import ConfigBaseModel
from topollm.config_classes.data.data_config import DataConfig
from topollm.config_classes.data_processing_column_names.data_processing_column_names import DataProcessingColumnNames
from topollm.config_classes.embeddings.embeddings_config import EmbeddingsConfig
from topollm.config_classes.embeddings_data_prep.embeddings_data_prep_config import EmbeddingsDataPrepConfig
from topollm.config_classes.feature_flags.feature_flags_config import FeatureFlagsConfig
from topollm.config_classes.finetuning.finetuning_config import FinetuningConfig
from topollm.config_classes.inference.inference_config import InferenceConfig
from topollm.config_classes.language_model.language_model_config import (
    LanguageModelConfig,
)
from topollm.config_classes.local_estimates.local_estimates_config import LocalEstimatesConfig
from topollm.config_classes.paths.paths_config import PathsConfig
from topollm.config_classes.storage.storage_config import StorageConfig
from topollm.config_classes.tokenizer.tokenizer_config import TokenizerConfig
from topollm.config_classes.transformations.transformations_config import TransformationsConfig
from topollm.config_classes.wandb.wandb_config import WandBConfig
from topollm.typing.enums import PreferredTorchBackend, Verbosity


class ComparisonDataConfig(ConfigBaseModel):
    """Config to specify what needs to be modified to pick out the comparison data."""

    embeddings: EmbeddingsConfig = Field(
        default_factory=EmbeddingsConfig,
        title="Embeddings configuration.",
        description="The configuration for specifying embeddings.",
    )

    local_estimates: LocalEstimatesConfig = Field(
        default=LocalEstimatesConfig(),
        title="Local estimates configuration.",
        description="The configuration for specifying local estimates.",
    )


class MainConfig(ConfigBaseModel):
    """Main configuration for all scripts."""

    analysis: AnalysisConfig = Field(
        default_factory=AnalysisConfig,
        title="Analysis configuration.",
        description="The configuration for specifying analysis parameters.",
    )

    comparison_data: ComparisonDataConfig = Field(
        default_factory=ComparisonDataConfig,
        title="Comparison data configuration.",
        description="The configuration for specifying comparison data.",
    )

    data: DataConfig = Field(
        default=...,
        title="Data configuration.",
        description="The configuration for specifying data.",
    )

    data_processing_column_names: DataProcessingColumnNames = Field(
        default_factory=DataProcessingColumnNames,
        title="Data processing column names.",
        description="The column names for data processing.",
    )

    embeddings_data_prep: EmbeddingsDataPrepConfig = Field(
        default=...,
        title="Embeddings data preparation configuration.",
        description="The configuration for specifying embeddings data preparation.",
    )

    embeddings: EmbeddingsConfig = Field(
        default_factory=EmbeddingsConfig,
        title="Embeddings configuration.",
        description="The configuration for specifying embeddings.",
    )

    feature_flags: FeatureFlagsConfig = Field(
        default_factory=FeatureFlagsConfig,
        title="Feature flags configuration.",
        description="The configuration for specifying feature flags.",
    )

    finetuning: FinetuningConfig = Field(
        default=...,
        title="Finetuning configuration.",
        description="The configuration for specifying finetuning.",
    )

    inference: InferenceConfig = Field(
        default=...,
        title="Inference configuration.",
        description="The configuration for specifying inference.",
    )

    language_model: LanguageModelConfig = Field(
        default=...,
        title="Model configuration.",
        description="The configuration for specifying model.",
    )

    local_estimates: LocalEstimatesConfig = Field(
        default=LocalEstimatesConfig(),
        title="Local estimates configuration.",
        description="The configuration for specifying local estimates.",
    )

    paths: PathsConfig = Field(
        default=PathsConfig(),
        title="Paths configuration.",
        description="The configuration for specifying paths.",
    )

    preferred_torch_backend: PreferredTorchBackend = Field(
        default=PreferredTorchBackend.CPU,
        title="Preferred torch backend.",
        description="The preferred torch backend.",
    )

    global_seed: int = Field(
        default=1234,
        title="Global seed.",
        description="Global seed. Currently the global seed is rarely used, "
        "since we have specific seeds for the individual sampling steps and for the finetuning.",
    )

    storage: StorageConfig = Field(
        default=StorageConfig(),
        title="Storage configuration.",
        description="The configuration for specifying storage.",
    )

    tokenizer: TokenizerConfig = Field(
        default=...,
        title="Tokenizer configuration.",
        description="The configuration for specifying tokenizer.",
    )

    transformations: TransformationsConfig = Field(
        default_factory=TransformationsConfig,
        title="Transformations configuration.",
        description="The configuration for specifying transformations.",
    )

    verbosity: Verbosity = Field(
        default=Verbosity.NORMAL,
        title="Verbosity level.",
        description="The verbosity level.",
    )

    n_jobs: int = Field(
        default=1,
        title="Number of jobs.",
        description=(
            "The number of jobs to use for scripts which support multiprocessing. "
            "If 1 is given, no parallel computing code will be used at all, which is useful for debugging."
        ),
    )

    wandb: WandBConfig = Field(
        default_factory=WandBConfig,
        title="Weights and Biases configuration.",
        description="The configuration for specifying Weights and Biases.",
    )

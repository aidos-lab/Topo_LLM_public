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

"""Configuration class for feature flags."""

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel


class CompareSamplingMethodsFeatureFlagsConfig(ConfigBaseModel):
    """Feature flags for the comparison of sampling methods."""

    do_iterate_all_partial_search_base_directories: bool = Field(
        default=True,
        title="Iterate over all partial search base directories.",
        description="Whether to iterate over all partial search base directories.",
    )

    do_analysis_influence_of_local_estimates_n_neighbors: bool = Field(
        default=True,
        title="Analyze the influence of the number of neighbors in the local estimates.",
        description="Whether to analyze the influence of the number of neighbors in the local estimates.",
    )

    do_create_boxplot_of_mean_over_different_sampling_seeds: bool = Field(
        default=True,
        title="Create boxplot of mean over different sampling seeds.",
        description="Whether to create a boxplot of the mean over different sampling seeds.",
    )


class AnalysisFeatureFlagsConfig(ConfigBaseModel):
    """Feature flags for the analysis process."""

    create_plots_in_local_estimates_worker: bool = Field(
        default=True,
        title="Create plots in local estimates worker.",
        description="Whether to create plots in the local estimates worker.",
    )

    compare_sampling_methods: CompareSamplingMethodsFeatureFlagsConfig = Field(
        default_factory=CompareSamplingMethodsFeatureFlagsConfig,
        title="Compare sampling methods.",
        description="Feature flags for the comparison of sampling methods.",
    )


class ComputeAndStoreEmbeddingsFeatureFlagsConfig(ConfigBaseModel):
    """Feature flags for the compute and store embeddings process."""

    skip_compute_and_store_embeddings: bool = Field(
        default=False,
        title="Skip compute and store embeddings in the pipeline.",
        description="Whether to skip the compute and store embeddings process in the pipeline.",
    )


class EmbeddingsDataPrepFeatureFlagsConfig(ConfigBaseModel):
    """Feature flags for the embeddings data preparation process."""

    add_additional_metadata: bool = Field(
        default=True,
        title="Add additional metadata to the saved metadata.",
        description="Whether to add additional metadata to the saved metadata. "
        "This needs to be enabled for the other flags to take effect.",
    )

    write_tokens_list_to_meta: bool = Field(
        default=True,
        title="Write tokens_list to the metadata "
        "(i.e., the list of tokens in the sequence/sentence in which a token appears).",
    )

    write_concatenated_tokens_to_meta: bool = Field(
        default=True,
        title="Write concatenated_tokens to the metadata (i.e., the sequence/sentence string in which a token appears).",
    )


class FinetuningFeatureFlagsConfig(ConfigBaseModel):
    """Feature flags for the finetuning process."""

    do_create_finetuned_language_model_config: bool = Field(
        default=True,
        title="Create finetuned language model config.",
        description="Whether to create the finetuned language model config after the finetuning process.",
    )

    skip_finetuning: bool = Field(
        default=False,
        title="Skip finetuning.",
        description="Whether to skip the finetuning process.",
    )


class ScriptsFeatureFlagsConfig(ConfigBaseModel):
    """Feature flags for the scripts."""

    dry_run: bool = Field(
        default=False,
        title="Use dry run mode.",
        description="Whether to use dry run mode.",
    )


class WandbFeatureFlagsConfig(ConfigBaseModel):
    """Feature flags for the Weights and Biases integration."""

    use_wandb: bool = Field(
        default=True,
        title="Use wandb.",
        description="Whether to use wandb for logging.",
    )


class FeatureFlagsConfig(ConfigBaseModel):
    """Configurations for specifying feature flags."""

    analysis: AnalysisFeatureFlagsConfig = Field(
        default_factory=AnalysisFeatureFlagsConfig,
        title="Analysis feature flags.",
        description="Feature flags for the analysis process.",
    )

    compute_and_store_embeddings: ComputeAndStoreEmbeddingsFeatureFlagsConfig = Field(
        default_factory=ComputeAndStoreEmbeddingsFeatureFlagsConfig,
        title="Compute and store embeddings feature flags.",
        description="Feature flags for the compute and store embeddings process.",
    )

    embeddings_data_prep: EmbeddingsDataPrepFeatureFlagsConfig = Field(
        default_factory=EmbeddingsDataPrepFeatureFlagsConfig,
        title="Embeddings data preparation feature flags.",
        description="Feature flags for the embeddings data preparation process.",
    )

    finetuning: FinetuningFeatureFlagsConfig = Field(
        default_factory=FinetuningFeatureFlagsConfig,
        title="Finetuning feature flags.",
        description="Feature flags for the finetuning process.",
    )

    scripts: ScriptsFeatureFlagsConfig = Field(
        default_factory=ScriptsFeatureFlagsConfig,
        title="Scripts feature flags.",
        description="Feature flags for the scripts.",
    )

    wandb: WandbFeatureFlagsConfig = Field(
        default_factory=WandbFeatureFlagsConfig,
        title="Weights and Biases feature flags.",
        description="Feature flags for the Weights and Biases integration.",
    )

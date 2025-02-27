# Copyright 2024-2025
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (mail@ruppik.net)
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

    # # # #
    # Different types of analysis

    do_noise_analysis: bool = Field(
        default=True,
        title="Do noise analysis.",
        description="Whether to do noise analysis.",
    )

    do_checkpoint_analysis: bool = Field(
        default=True,
        title="Do checkpoint analysis.",
        description="Whether to do checkpoint analysis.",
    )

    do_data_subsampling_number_of_samples_analysis: bool = Field(
        default=True,
        title="Do data subsampling number of samples analysis.",
        description="Whether to do data subsampling number of samples analysis.",
    )


class DistanceFunctionsFeatureFlagsConfig(ConfigBaseModel):
    """Feature flags for the distance functions."""

    use_exact_hausdorff: bool = Field(
        default=False,  # We set this to false by default, since the computation of the exact Hausdorff distance is very slow.
        title="Use exact Hausdorff distance.",
        description="Whether to use the exact Hausdorff distance.",
    )

    use_approximate_hausdorff_via_kdtree: bool = Field(
        default=True,
        title="Use approximate Hausdorff distance via KDTree.",
        description="Whether to use the approximate Hausdorff distance via KDTree.",
    )

    use_sinkhorn_wasserstein: bool = Field(
        default=False,  # We set this to false by default, since the computation with the `geomloss` has some issues on the cluster.
        title="Use Sinkhorn Wasserstein distance.",
        description="Whether to use the Sinkhorn Wasserstein distance.",
    )


class LocalEstimatesSavingFeatureFlagsConfig(ConfigBaseModel):
    """Feature flags for saving of the local estimates."""

    save_array_for_estimator: bool = Field(
        default=True,
        title="Save array for estimator.",
        description=("Whether to save the array for the estimator. Note that this might require a lot of disk space."),
    )


class AnalysisFeatureFlagsConfig(ConfigBaseModel):
    """Feature flags for the analysis process."""

    create_plots_in_local_estimates_worker: bool = Field(
        default=False,
        title="Create plots in local estimates worker.",
        description="Whether to create plots in the local estimates worker.",
    )

    compare_sampling_methods: CompareSamplingMethodsFeatureFlagsConfig = Field(
        default_factory=CompareSamplingMethodsFeatureFlagsConfig,
        title="Compare sampling methods.",
        description="Feature flags for the comparison of sampling methods.",
    )

    distance_functions: DistanceFunctionsFeatureFlagsConfig = Field(
        default_factory=DistanceFunctionsFeatureFlagsConfig,
        title="Distance functions.",
        description="Feature flags for the distance functions.",
    )

    saving: LocalEstimatesSavingFeatureFlagsConfig = Field(
        default_factory=LocalEstimatesSavingFeatureFlagsConfig,
        title="Local estimates saving.",
        description="Feature flags for saving of the local estimates.",
    )


class ComparisonFeatureFlagsConfig(ConfigBaseModel):
    """Feature flags for the comparison process."""

    do_comparison_of_local_estimates_with_losses: bool = Field(
        default=True,
        title="Do comparison of local estimates with losses.",
        description="Whether to do the comparison of local estimates with losses.",
    )

    do_comparison_of_local_estimates_between_base_data_and_comparison_data: bool = Field(
        default=True,
        title="Do comparison of local estimates between base and comparison data.",
        description="Whether to do the comparison of local estimates between base and comparison data.",
    )


class ComputeAndStoreEmbeddingsFeatureFlagsConfig(ConfigBaseModel):
    """Feature flags for the compute and store embeddings process."""

    skip_compute_and_store_embeddings_in_pipeline: bool = Field(
        default=False,
        title="Skip compute and store embeddings in the pipeline.",
        description="Whether to skip the compute and store embeddings process in the pipeline.",
    )


class EmbeddingsDataPrepFeatureFlagsConfig(ConfigBaseModel):
    """Feature flags for the embeddings data preparation process."""

    skip_embeddings_data_prep_in_pipeline: bool = Field(
        default=False,
        title="Skip embeddings data preparation in the pipeline.",
        description="Whether to skip the embeddings data preparation in the pipeline.",
    )

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


class TaskPerformanceAnalysisFeatureFlagsConfig(ConfigBaseModel):
    """Feature flags for the task performance analysis."""

    plotting_create_distribution_plots_over_model_checkpoints: bool = Field(
        default=True,
        title="Create distribution plots over model checkpoints.",
        description="Whether to create distribution plots over model checkpoints.",
    )

    plotting_create_distribution_plots_over_model_layers: bool = Field(
        default=True,
        title="Create distribution plots over model layers.",
        description="Whether to create distribution plots over model layers.",
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

    comparison: ComparisonFeatureFlagsConfig = Field(
        default_factory=ComparisonFeatureFlagsConfig,
        title="Comparison feature flags.",
        description="Feature flags for the comparison process.",
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

    task_performance_analysis: TaskPerformanceAnalysisFeatureFlagsConfig = Field(
        default_factory=TaskPerformanceAnalysisFeatureFlagsConfig,
        title="Task performance analysis feature flags.",
        description="Feature flags for the task performance analysis.",
    )

    wandb: WandbFeatureFlagsConfig = Field(
        default_factory=WandbFeatureFlagsConfig,
        title="Weights and Biases feature flags.",
        description="Feature flags for the Weights and Biases integration.",
    )

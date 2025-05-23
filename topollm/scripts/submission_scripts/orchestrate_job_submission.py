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

"""Orchestrate job submission based on a specific experiment setup."""

import click

from topollm.scripts.submission_scripts.make_submission_config_and_run_task import (
    make_machine_config,
    make_submission_config_and_run_task,
)
from topollm.scripts.submission_scripts.submission_config import MachineConfig, Template
from topollm.scripts.submission_scripts.types import (
    CheckpointNoListOption,
    DataListOption,
    DataSubsamplingNumberOfSamplesListOption,
    DataSubsamplingSamplingSeedListOption,
    EmbeddingsDataPrepNumSamplesListOption,
    EmbeddingsDataPrepSamplingSeedListOption,
    ExperimentSelector,
    ExperimentStage,
    FinetuningBaseModelListOption,
    FinetuningDatasetsListOption,
    FinetuningRegimeOption,
    LanguageModelListOption,
    LocalEstimatesFilteringNumSamplesListOption,
    LocalEstimatesPointwiseAbsoluteNNeighborsListOption,
    ModelGroupOption,
    RunOnlySelectedConfigsOption,
    RunOption,
    SeedListOption,
)
from topollm.typing.enums import (
    DataSamplingMode,
    EmbeddingDataHandlerMode,
    EmbeddingsDataPrepSamplingMode,
    SubmissionMode,
    Task,
)


@click.command()
@click.option(
    "--experiment-stage",
    type=click.Choice(
        choices=list(ExperimentStage),
    ),
    required=False,
    default=None,
    help="Specify the experiment stage to run.",
)
@click.option(
    "--experiment-selector",
    type=click.Choice(
        choices=list(ExperimentSelector),
    ),
    help="Select the experiment type.",
)
@click.option(
    "--task",
    type=click.Choice(
        choices=list(Task),
    ),
    required=True,
    help="Specify the task to run.",
)
@click.option(
    "--data-list-option",
    type=click.Choice(
        choices=list(DataListOption),
    ),
    default=DataListOption.MULTIWOZ21_ONLY,
    help="Data list option to use.",
)
@click.option(
    "--data-subsampling-sampling-seed-list-option",
    type=click.Choice(
        choices=list(DataSubsamplingSamplingSeedListOption),
    ),
    default=DataSubsamplingSamplingSeedListOption.THREE_SEEDS,
    help="Data subsampling sampling seed list option to use.",
)
@click.option(
    "--data-subsampling-sampling-mode",
    type=click.Choice(
        choices=list(DataSamplingMode),
    ),
    default=DataSamplingMode.RANDOM,
    help="Data subsampling sampling mode to use.",
)
@click.option(
    "--embedding-data-handler-mode",
    type=click.Choice(
        choices=list(EmbeddingDataHandlerMode),
    ),
    default=EmbeddingDataHandlerMode.REGULAR,
    help="Embedding data handler mode to use.",
)
@click.option(
    "--embeddings-data-prep-sampling-mode",
    type=click.Choice(
        choices=list(EmbeddingsDataPrepSamplingMode),
    ),
    default=EmbeddingsDataPrepSamplingMode.RANDOM,
    help="Embeddings data prep sampling mode to use.",
)
@click.option(
    "--model-group-option",
    type=click.Choice(
        choices=list(ModelGroupOption),
    ),
    default=ModelGroupOption.ROBERTA_BASE_WITHOUT_MODIFICATIONS,
    help="The model group and finetuning regime to use.",
)
@click.option(
    "--finetuning-base-model-list-option",
    type=click.Choice(
        choices=list(FinetuningBaseModelListOption),
    ),
    default=FinetuningBaseModelListOption.ROBERTA_BASE,
    help="Finetuning base model list option to use.",
)
@click.option(
    "--finetuning-datasets-list-option",
    type=click.Choice(
        choices=list(FinetuningDatasetsListOption),
    ),
    default=FinetuningDatasetsListOption.MULTIWOZ21_SMALL,
    help="Finetuning datasets list option to use.",
)
@click.option(
    "--fp16",
    type=str,
    default="true",
    help="Whether to use FP16.",
)
@click.option(
    "--local-estimates-pointwise-absolute-n-neighbors-list-option",
    type=click.Choice(
        choices=list(LocalEstimatesPointwiseAbsoluteNNeighborsListOption),
    ),
    default=LocalEstimatesPointwiseAbsoluteNNeighborsListOption.SINGLE_CHOICE_128,
    help="Local estimates pointwise absolute n neighbors list option to use.",
)
@click.option(
    "--wandb-project",
    type=str,
    default="Topo_LLM_finetuning_from_submission_script",
    help="Wandb project to use.",
)
@click.option(
    "--additional-overrides",
    type=str,
    multiple=True,
)
@click.option(
    "--submission-mode",
    type=click.Choice(
        choices=list(SubmissionMode),
    ),
    default=SubmissionMode.HPC_SUBMISSION,
    help="Whether to run the job on the HPC or locally.",
)
@click.option(
    "--run-option",
    type=click.Choice(
        choices=list(RunOption),
    ),
    default=RunOption.DO_SUBMISSION,
    help="Whether to do the submission or start a dry run.",
)
@click.option(
    "--run-only-selected-configs-option",
    type=click.Choice(
        choices=list(RunOnlySelectedConfigsOption),
    ),
    default=RunOnlySelectedConfigsOption.RUN_ALL,
    help="Run only a selected set of configurations.",
)
@click.option(
    "--memory",
    type=str,
    default="32",
    help="Amount of memory to allocate.",
)
@click.option(
    "--ncpus",
    type=str,
    default="2",
    help="Number of CPUs to allocate.",
)
@click.option(
    "--ngpus",
    type=str,
    default="1",
    help="Number of GPUs to allocate.",
)
@click.option(
    "--queue",
    type=str,
    default="DSML",
    help="Queue to submit the job to.",
)
@click.option(
    "--template",
    type=click.Choice(
        choices=list(Template),
    ),
    default=Template.DSML,
    help="Template to use for the job submission. Might get overwritten by the experiment stage configurations.",
)
@click.option(
    "--template-to-use-for-compute-embeddings",
    type=click.Choice(
        choices=list(Template),
    ),
    default=Template.RTX6000,
    help="Template to use for the compute embeddings job submission.",
)
def orchestrate_job_submission(
    experiment_stage: ExperimentStage | None,
    experiment_selector: ExperimentSelector,
    task: Task,
    model_group_option: ModelGroupOption,
    finetuning_base_model_list_option: FinetuningBaseModelListOption,
    finetuning_datasets_list_option: FinetuningDatasetsListOption,
    fp16: str,
    data_list_option: DataListOption,
    data_subsampling_sampling_mode: DataSamplingMode,
    data_subsampling_sampling_seed_list_option: DataSubsamplingSamplingSeedListOption,
    embedding_data_handler_mode: EmbeddingDataHandlerMode,
    embeddings_data_prep_sampling_mode: EmbeddingsDataPrepSamplingMode,
    local_estimates_pointwise_absolute_n_neighbors_list_option: LocalEstimatesPointwiseAbsoluteNNeighborsListOption,
    wandb_project: str,
    *,
    memory: str,
    ncpus: str,
    ngpus: str,
    queue: str,
    template: Template,
    template_to_use_for_compute_embeddings: Template,
    additional_overrides: list[str] | None,
    submission_mode: SubmissionMode,
    run_option: RunOption,
    run_only_selected_configs_option: RunOnlySelectedConfigsOption,
) -> None:
    """Submit jobs based on the specified options."""
    ########################################
    # Model-specific configurations
    #
    # Note:
    # - For the finetuning task, the finetuning_regime_option might get overwritten at a later stage.
    match model_group_option:
        # # # # # # # # # # # #
        # RoBERTa-base models
        case ModelGroupOption.ROBERTA_BASE_WITHOUT_MODIFICATIONS:
            ####################################
            ### With POS tags for base model ###
            language_model_list_option = LanguageModelListOption.ROBERTA_BASE
            finetuning_regime_option = FinetuningRegimeOption.FEW_EPOCHS  # Ignored for the base model
            language_model_seed_list_option = SeedListOption.DO_NOT_SET
            checkpoint_no_list_option = CheckpointNoListOption.SELECTED  # Ignored for the base model
        case ModelGroupOption.ROBERTA_BASE_FINETUNED_FOR_FEW_EPOCHS_OLD_AND_NEW_DATA_SINGLE_SEED_LAST_CHECKPOINT:
            ################################################################
            language_model_list_option = (
                LanguageModelListOption.FINETUNED_ON_OLD_AND_NEW_DATA_FEW_EPOCHS_FROM_ROBERTA_BASE
            )
            finetuning_regime_option = FinetuningRegimeOption.FEW_EPOCHS
            language_model_seed_list_option = SeedListOption.FIXED_SEED_1234
            checkpoint_no_list_option = CheckpointNoListOption.FIXED_2800
        case ModelGroupOption.ROBERTA_BASE_FINETUNED_FOR_FEW_EPOCHS_MULTIWOZ_DATA_SINGLE_SEED_LAST_CHECKPOINT:
            language_model_list_option = LanguageModelListOption.FINETUNED_ON_MULTIWOZ_DATA_FEW_EPOCHS_FROM_ROBERTA_BASE
            finetuning_regime_option = FinetuningRegimeOption.FEW_EPOCHS
            language_model_seed_list_option = SeedListOption.FIXED_SEED_1234
            checkpoint_no_list_option = CheckpointNoListOption.FIXED_2800
        case ModelGroupOption.ROBERTA_BASE_FINETUNED_FOR_FEW_EPOCHS_MULTIWOZ_DATA_SINGLE_SEED_ALL_CHECKPOINTS_STEP_100:
            language_model_list_option = LanguageModelListOption.FINETUNED_ON_MULTIWOZ_DATA_FEW_EPOCHS_FROM_ROBERTA_BASE
            finetuning_regime_option = FinetuningRegimeOption.FEW_EPOCHS
            language_model_seed_list_option = SeedListOption.FIXED_SEED_1234
            checkpoint_no_list_option = CheckpointNoListOption.RANGE_START_100_STOP_3200_STEP_100
        case ModelGroupOption.ROBERTA_BASE_FINETUNED_FOR_FEW_EPOCHS_MULTIWOZ_DATA_SINGLE_SEED_FROZEN_LM_HEAD_ALL_CHECKPOINTS_STEP_100:
            language_model_list_option = (
                LanguageModelListOption.FINETUNED_ON_MULTIWOZ_DATA_FEW_EPOCHS_FROZEN_LM_HEAD_FROM_ROBERTA_BASE
            )
            finetuning_regime_option = FinetuningRegimeOption.FEW_EPOCHS
            language_model_seed_list_option = SeedListOption.FIXED_SEED_1234
            checkpoint_no_list_option = CheckpointNoListOption.RANGE_START_100_STOP_3200_STEP_100
        case ModelGroupOption.ROBERTA_BASE_FINETUNED_FOR_FEW_EPOCHS_MULTIWOZ_AND_REDDIT_AND_WIKITEXT_DATA_SINGLE_SEED_ALL_CHECKPOINTS_STEP_400:
            language_model_list_option = (
                LanguageModelListOption.FINETUNED_ON_MULTIWOZ_AND_REDDIT_AND_WIKITEXT_DATA_FEW_EPOCHS_FROM_ROBERTA_BASE
            )
            finetuning_regime_option = FinetuningRegimeOption.FEW_EPOCHS
            language_model_seed_list_option = SeedListOption.FIXED_SEED_1234
            checkpoint_no_list_option = CheckpointNoListOption.RANGE_START_400_STOP_3200_STEP_400
        case ModelGroupOption.ROBERTA_BASE_FINETUNED_FOR_MANY_EPOCHS:
            ################################################################
            ### With POS tags for finetuned models and three checkpoints ###
            language_model_list_option = LanguageModelListOption.SELECTED_FINETUNED_MANY_EPOCHS_FROM_ROBERTA_BASE
            finetuning_regime_option = FinetuningRegimeOption.MANY_EPOCHS_WITH_OVERFITTING_RISK
            language_model_seed_list_option = SeedListOption.FIXED_SEED_1234
            checkpoint_no_list_option = CheckpointNoListOption.ONLY_BEGINNING_AND_MIDDLE_AND_END
        # # # # # # # # # # # #
        # GPT-2 models
        #
        # Notes:
        # - Remember to add new model in the case distinctions in the `make_machine_config` function,
        #   if they require different memory sizes.
        case ModelGroupOption.GPT2_MEDIUM_WITHOUT_MODIFICATIONS:
            language_model_list_option = LanguageModelListOption.GPT2_MEDIUM
            finetuning_regime_option = FinetuningRegimeOption.FEW_EPOCHS  # Ignored for the base model
            language_model_seed_list_option = SeedListOption.DO_NOT_SET
            checkpoint_no_list_option = CheckpointNoListOption.SELECTED  # Ignored for the base model
        case ModelGroupOption.GPT2_MEDIUM_FINETUNED_FOR_FEW_EPOCHS_MULTIWOZ_AND_REDDIT_AND_WIKITEXT_DATA_SINGLE_SEED_LAST_CHECKPOINT:
            language_model_list_option = (
                LanguageModelListOption.FINETUNED_ON_MULTIWOZ_AND_REDDIT_AND_WIKITEXT_DATA_FEW_EPOCHS_FROM_GPT2_MEDIUM
            )
            finetuning_regime_option = FinetuningRegimeOption.FEW_EPOCHS
            language_model_seed_list_option = SeedListOption.FIXED_SEED_1234
            checkpoint_no_list_option = CheckpointNoListOption.FIXED_2800
        case ModelGroupOption.GPT2_MEDIUM_FINETUNED_FOR_FEW_EPOCHS_MULTIWOZ_AND_REDDIT_AND_WIKITEXT_DATA_SINGLE_SEED_CHECKPOINTS_1200_1600:
            language_model_list_option = (
                LanguageModelListOption.FINETUNED_ON_MULTIWOZ_AND_REDDIT_AND_WIKITEXT_DATA_FEW_EPOCHS_FROM_GPT2_MEDIUM
            )
            finetuning_regime_option = FinetuningRegimeOption.FEW_EPOCHS
            language_model_seed_list_option = SeedListOption.FIXED_SEED_1234
            checkpoint_no_list_option = CheckpointNoListOption.FIXED_1200_1600
        case ModelGroupOption.GPT2_MEDIUM_FINETUNED_FOR_FEW_EPOCHS_WIKITEXT_DATA_SINGLE_SEED_CHECKPOINTS_1200_1600:
            language_model_list_option = LanguageModelListOption.FINETUNED_ON_WIKITEXT_DATA_FEW_EPOCHS_FROM_GPT2_MEDIUM
            finetuning_regime_option = FinetuningRegimeOption.FEW_EPOCHS
            language_model_seed_list_option = SeedListOption.FIXED_SEED_1234
            checkpoint_no_list_option = CheckpointNoListOption.FIXED_1200_1600
        case _:
            msg: str = f"Unknown {model_group_option = }"
            raise ValueError(
                msg,
            )

    ########################################
    ### Default configurations
    ###
    ### Note that these values might be overridden by the individual experiment setup below.

    add_prefix_space = False  # Note: Make sure that this `add_prefix_space` is consistent with the fine-tuning setup
    create_pos_tags = False
    skip_compute_and_store_embeddings_in_pipeline = False
    skip_embeddings_data_prep_in_pipeline = False

    # `embeddings_data_prep_sampling_seed_list_option` is set here and will be overwritten
    # in the experiment stage configurations below.
    embeddings_data_prep_sampling_seed_list_option = EmbeddingsDataPrepSamplingSeedListOption.FIVE_SEEDS

    embeddings_data_prep_num_samples_list_option = EmbeddingsDataPrepNumSamplesListOption.SINGLE_CHOICE_150000

    layer_indices_list: list[str] = [
        "[-1]",
    ]

    local_estimates_filtering_num_samples_list_option = LocalEstimatesFilteringNumSamplesListOption.SINGLE_CHOICE_60000

    # Notes on memory size:
    #
    # ++ accelerator_model=rtx6000:
    #   + `--common_batch_size="32"` appears to work for fine-tuning "roberta-base" model on rtx6000 with 24GB of VRAM.
    #
    # - Note that some previous fine-tuning runs were done with a batch size of 8.
    common_batch_size = 8
    batch_size_train = common_batch_size
    batch_size_eval = common_batch_size

    finetuning_seed_list_option = SeedListOption.ONE_SEED

    ########################################
    ### Experiment stage configurations
    ###
    ### Note that this might be overridden by the experiment selector configurations below.
    ########################################
    match experiment_stage:
        case ExperimentStage.COMPUTE_EMBEDDINGS_PLUS_SINGLE_PIPELINE_RUN:
            # Only run for a single embeddings data prep sampling seed
            embeddings_data_prep_sampling_seed_list_option = EmbeddingsDataPrepSamplingSeedListOption.DEFAULT
            skip_compute_and_store_embeddings_in_pipeline = False  # do the embeddings computation
        case ExperimentStage.SKIP_COMPUTE_EMBEDDINGS_BUT_DO_MULTIPLE_PIPELINE_RUNS:
            # Assume embeddings are already computed and run for different embeddings data prep sampling seeds
            embeddings_data_prep_sampling_seed_list_option = EmbeddingsDataPrepSamplingSeedListOption.FIVE_SEEDS
            skip_compute_and_store_embeddings_in_pipeline = True  # skip the embeddings computation
        case ExperimentStage.SKIP_COMPUTE_EMBEDDINGS_AND_SKIP_EMBEDDINGS_DATA_PREP:
            # Assume embeddings are already computed and skip the embeddings data prep step
            skip_compute_and_store_embeddings_in_pipeline = True  # skip the embeddings computation
            skip_embeddings_data_prep_in_pipeline = True  # skip the embeddings data prep step
        case _:
            msg: str = f"Unknown {experiment_stage = }"
            raise ValueError(
                msg,
            )

    ########################################
    ### Experiment selector configurations
    ###
    ### Note: You can use the experiment selector to override the default configurations above.
    ########################################
    match experiment_selector:
        #
        # >>> START Sensitivity analysis experiments
        #
        case ExperimentSelector.SENSITIVITY_ANALYSIS_MULTIWOZ21_DIFFERENT_DATA_SUBSAMPLING_NUMBER_OF_SAMPLES:
            # ++++ Experiment > different subsampling number of samples for multiwoz21 dataset
            #
            # Note:
            # - There are different setups for the multiwoz21 and the reddit dataset,
            #   since they have a different number of samples.
            data_list_option = DataListOption.MULTIWOZ21_ONLY
            data_subsampling_number_of_samples_list_option = (
                DataSubsamplingNumberOfSamplesListOption.RANGE_START_2000_STOP_18000_STEP_2000
            )

            data_subsampling_sampling_seed_list_option = DataSubsamplingSamplingSeedListOption.FIVE_SEEDS
        case ExperimentSelector.SENSITIVITY_ANALYSIS_REDDIT_DIFFERENT_DATA_SUBSAMPLING_NUMBER_OF_SAMPLES:
            # ++++ Experiment > different subsampling number of samples for reddit dataset
            #
            # Note:
            # - There are different setups for the multiwoz21 and the reddit dataset,
            #   since they have a different number of samples.
            # - We explicitly increase the memory size here,
            #   since for the embeddings data prep step on 12_000 and more data subsamlping samples,
            #   the embeddings data prep step requires more memory.
            data_list_option = DataListOption.REDDIT_ONLY
            data_subsampling_number_of_samples_list_option = (
                DataSubsamplingNumberOfSamplesListOption.RANGE_START_2000_STOP_24000_STEP_2000
            )

            data_subsampling_sampling_seed_list_option = DataSubsamplingSamplingSeedListOption.FIVE_SEEDS
        case ExperimentSelector.SENSITIVITY_ANALYSIS_DIFFERENT_LOCAL_ESTIMATES_FILTERING_NUMBER_OF_SAMPLES:
            # Notes:
            # - You need to set the data_list_option via the command line arguments.
            # - Do not set the checkpoint_no_list_option here, since we want to take it from the model group option.

            # Sequence subsampling: Fixed
            data_subsampling_number_of_samples_list_option = DataSubsamplingNumberOfSamplesListOption.FIXED_10000
            data_subsampling_sampling_seed_list_option = DataSubsamplingSamplingSeedListOption.FIXED_777

            # Token subsampling:
            # - Different sampling seeds
            # - Different local estimates number of samples
            embeddings_data_prep_sampling_seed_list_option = EmbeddingsDataPrepSamplingSeedListOption.FIVE_SEEDS
            local_estimates_filtering_num_samples_list_option = (
                LocalEstimatesFilteringNumSamplesListOption.RANGE_START_10000_STOP_110000_STEP_10000
            )

            embedding_data_handler_mode = EmbeddingDataHandlerMode.REGULAR
        case ExperimentSelector.SENSITIVITY_ANALYSIS_DIFFERENT_LOCAL_ESTIMATES_POINTWISE_ABSOLUTE_N_NEIGHBORS:
            # Notes:
            # - You need to set the data_list_option via the command line arguments.
            # - Do not set the checkpoint_no_list_option here, since we want to take it from the model group option.

            data_subsampling_number_of_samples_list_option = DataSubsamplingNumberOfSamplesListOption.FIXED_10000
            data_subsampling_sampling_seed_list_option = DataSubsamplingSamplingSeedListOption.FIXED_777

            # Note: We currently run this for a single token sampling seed,
            # to reduce the number of runs.
            embeddings_data_prep_sampling_seed_list_option = EmbeddingsDataPrepSamplingSeedListOption.DEFAULT

            embedding_data_handler_mode = EmbeddingDataHandlerMode.REGULAR

            local_estimates_pointwise_absolute_n_neighbors_list_option = (
                LocalEstimatesPointwiseAbsoluteNNeighborsListOption.POWERS_OF_TWO_UP_TO_1024
            )
        #
        # >>> END Sensitivity analysis experiments
        #
        case ExperimentSelector.COARSE_CHECKPOINT_RESOLUTION:
            # ++++ Experiment > Coarse checkpoint resolution
            #
            # Notes:
            # - You need to set the data_list_option via the command line arguments.
            data_subsampling_number_of_samples_list_option = DataSubsamplingNumberOfSamplesListOption.FIXED_10000

            checkpoint_no_list_option = CheckpointNoListOption.SELECTED
        case ExperimentSelector.EXPLORATORY_DROPOUT_ANALYSIS_COARSE_CHECKPOINT_RESOLUTION:
            # ++++ Experiment > Coarse checkpoint resolution for first exploratory dropout analysis
            #
            # Notes:
            # - You need to set the data_list_option via the command line arguments.
            data_subsampling_number_of_samples_list_option = DataSubsamplingNumberOfSamplesListOption.FIXED_10000

            # Select a few of the dropout models for the first exploratory dropout analysis
            language_model_list_option = LanguageModelListOption.WITH_005_015_02_DROPOUT_FINETUNED_ON_MULTIWOZ_SMALL_MANY_EPOCHS_FROM_ROBERTA_BASE
            finetuning_regime_option = FinetuningRegimeOption.MANY_EPOCHS_WITH_OVERFITTING_RISK
            # Select only a single training seed
            language_model_seed_list_option = SeedListOption.FIXED_SEED_1234

            checkpoint_no_list_option = CheckpointNoListOption.SELECTED
        case ExperimentSelector.TINY_DROPOUT_VARIATIONS_COARSE_CHECKPOINT_RESOLUTION:
            # ++++ Experiment > Coarse checkpoint resolution for first dropout with small variations analysis
            #
            # Notes:
            # - You need to set the data_list_option via the command line arguments.
            data_subsampling_number_of_samples_list_option = DataSubsamplingNumberOfSamplesListOption.FIXED_10000

            # Select a few of the dropout models for the first exploratory dropout analysis
            language_model_list_option = (
                LanguageModelListOption.WITH_006_007_DROPOUT_FINETUNED_ON_MULTIWOZ_SMALL_MANY_EPOCHS_FROM_ROBERTA_BASE
            )
            finetuning_regime_option = FinetuningRegimeOption.MANY_EPOCHS_WITH_OVERFITTING_RISK
            # Select only a single training seed
            language_model_seed_list_option = SeedListOption.FIXED_SEED_1234

            checkpoint_no_list_option = CheckpointNoListOption.SELECTED
        case ExperimentSelector.FIXED_PARAMETERS_HIGH_CHECKPOINT_RESOLUTION:
            # ++++ Experiment > Fixing many of the parameters so that we can run the
            #      checkpoint comparison experiment with high checkpoint resolution
            #
            # Notes:
            # - You need to set the data_list_option via the command line arguments.
            data_subsampling_number_of_samples_list_option = DataSubsamplingNumberOfSamplesListOption.FIXED_10000

            # Uncomment the following to do this only for one data subsampling sampling seed
            data_subsampling_sampling_seed_list_option = DataSubsamplingSamplingSeedListOption.FIXED_777

            # Select the models which are fine-tuned until they run into overfitting
            language_model_list_option = LanguageModelListOption.SELECTED_FINETUNED_MANY_EPOCHS_FROM_ROBERTA_BASE
            finetuning_regime_option = FinetuningRegimeOption.MANY_EPOCHS_WITH_OVERFITTING_RISK
            # Select only a single training seed
            language_model_seed_list_option = SeedListOption.FIXED_SEED_1234

            # Select all checkpoints for which we have evaluation results
            checkpoint_no_list_option = CheckpointNoListOption.FULL
        case ExperimentSelector.REGULAR_TOKEN_EMBEDDINGS:
            # Notes:
            # - You need to set the data_list_option via the command line arguments.
            # - Do not set the checkpoint_no_list_option here, since we want to take it from the model group option.

            data_subsampling_number_of_samples_list_option = DataSubsamplingNumberOfSamplesListOption.FIXED_10000

            embedding_data_handler_mode = EmbeddingDataHandlerMode.REGULAR

            # Select only a single training seed
            language_model_seed_list_option = SeedListOption.FIXED_SEED_1234
        case ExperimentSelector.MASKED_TOKEN_EMBEDDINGS:
            # Notes:
            # - You need to set the data_list_option via the command line arguments.
            # - Do not set the checkpoint_no_list_option here, since we want to take it from the model group option.

            data_subsampling_number_of_samples_list_option = DataSubsamplingNumberOfSamplesListOption.FIXED_10000

            embedding_data_handler_mode = EmbeddingDataHandlerMode.MASKED_TOKEN

            # Select only a single training seed
            language_model_seed_list_option = SeedListOption.FIXED_SEED_1234
        case ExperimentSelector.REGULAR_TOKEN_EMBEDDINGS_MULTIPLE_LAYERS_SINGLE_SAMPLE:
            # Notes:
            # - You need to set the data_list_option via the command line arguments.
            # - Do not set the checkpoint_no_list_option here, since we want to take it from the model group option.

            data_subsampling_number_of_samples_list_option = DataSubsamplingNumberOfSamplesListOption.FIXED_10000
            data_subsampling_sampling_seed_list_option = DataSubsamplingSamplingSeedListOption.FIXED_777

            embedding_data_handler_mode = EmbeddingDataHandlerMode.REGULAR

            if language_model_list_option in [
                LanguageModelListOption.ROBERTA_BASE,
                LanguageModelListOption.FINETUNED_ON_OLD_AND_NEW_DATA_FEW_EPOCHS_FROM_ROBERTA_BASE,
                LanguageModelListOption.FINETUNED_ON_MULTIWOZ_DATA_FEW_EPOCHS_FROM_ROBERTA_BASE,
                LanguageModelListOption.FINETUNED_ON_MULTIWOZ_AND_REDDIT_AND_WIKITEXT_DATA_FEW_EPOCHS_FROM_ROBERTA_BASE,
                LanguageModelListOption.SELECTED_FINETUNED_FEW_EPOCHS_FROM_ROBERTA_BASE,
                LanguageModelListOption.SELECTED_FINETUNED_MANY_EPOCHS_FROM_ROBERTA_BASE,
                LanguageModelListOption.FULL_FINETUNED_FEW_EPOCHS_FROM_ROBERTA_BASE,
                LanguageModelListOption.WITH_005_015_02_DROPOUT_FINETUNED_ON_MULTIWOZ_SMALL_MANY_EPOCHS_FROM_ROBERTA_BASE,
                LanguageModelListOption.WITH_006_007_DROPOUT_FINETUNED_ON_MULTIWOZ_SMALL_MANY_EPOCHS_FROM_ROBERTA_BASE,
            ]:
                # Use the last 12 layers for models with roberta-base architecture
                layer_indices_list = [
                    "[-1]",
                    "[-2]",
                    "[-3]",
                    "[-4]",
                    "[-5]",
                    "[-6]",
                    "[-7]",
                    "[-8]",
                    "[-9]",
                    "[-10]",
                    "[-11]",
                    "[-12]",
                ]
            elif language_model_list_option in [
                LanguageModelListOption.GPT2_MEDIUM,
                LanguageModelListOption.FINETUNED_ON_OLD_AND_NEW_DATA_FEW_EPOCHS_FROM_GPT2_MEDIUM,
                LanguageModelListOption.FINETUNED_ON_MULTIWOZ_DATA_FEW_EPOCHS_FROM_GPT2_MEDIUM,
                LanguageModelListOption.FINETUNED_ON_WIKITEXT_DATA_FEW_EPOCHS_FROM_GPT2_MEDIUM,
                LanguageModelListOption.FINETUNED_ON_MULTIWOZ_AND_REDDIT_AND_WIKITEXT_DATA_FEW_EPOCHS_FROM_GPT2_MEDIUM,
            ]:
                # Use every second layer for models with gpt2-medium architecture
                layer_indices_list = [
                    "[-1]",
                    "[-3]",
                    "[-5]",
                    "[-7]",
                    "[-9]",
                    "[-11]",
                    "[-13]",
                    "[-15]",
                    "[-17]",
                    "[-19]",
                    "[-21]",
                    "[-23]",
                ]
            else:
                msg: str = f"Unsupported {language_model_list_option = } for the layerwise experiments."
                raise ValueError(
                    msg,
                )
        case ExperimentSelector.REGULAR_TOKEN_EMBEDDINGS_LAST_LAYER_SINGLE_SAMPLE:
            # Notes:
            # - You need to set the data_list_option via the command line arguments.
            # - Do not set the checkpoint_no_list_option here, since we want to take it from the model group option.

            data_subsampling_number_of_samples_list_option = DataSubsamplingNumberOfSamplesListOption.FIXED_10000
            data_subsampling_sampling_seed_list_option = DataSubsamplingSamplingSeedListOption.FIXED_777

            layer_indices_list = [
                "[-1]",
            ]

            embedding_data_handler_mode = EmbeddingDataHandlerMode.REGULAR
        case ExperimentSelector.MASKED_TOKEN_EMBEDDINGS_LAST_LAYER_SINGLE_SAMPLE:
            # Notes:
            # - You need to set the data_list_option via the command line arguments.
            # - Do not set the checkpoint_no_list_option here, since we want to take it from the model group option.

            data_subsampling_number_of_samples_list_option = DataSubsamplingNumberOfSamplesListOption.FIXED_10000
            data_subsampling_sampling_seed_list_option = DataSubsamplingSamplingSeedListOption.FIXED_777

            layer_indices_list = [
                "[-1]",
            ]

            embedding_data_handler_mode = EmbeddingDataHandlerMode.MASKED_TOKEN
        case ExperimentSelector.MASKED_TOKEN_EMBEDDINGS_LAST_LAYER_TWO_DATA_SUBSAMPLING_SAMPLING_SEEDS:
            # Notes:
            # - You need to set the data_list_option via the command line arguments.
            # - Do not set the checkpoint_no_list_option here, since we want to take it from the model group option.

            data_subsampling_number_of_samples_list_option = DataSubsamplingNumberOfSamplesListOption.FIXED_10000
            data_subsampling_sampling_seed_list_option = DataSubsamplingSamplingSeedListOption.FIXED_778_779

            layer_indices_list = [
                "[-1]",
            ]

            embedding_data_handler_mode = EmbeddingDataHandlerMode.MASKED_TOKEN
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # NOTE: You can add more experiment configurations here.
        case _:
            msg: str = f"Unknown {experiment_selector = }"
            raise click.UsageError(message=msg)

    ########################################
    ### Additional logic,
    ### for example to remove unnecessary configurations and thus avoid unnecessary computations
    ########################################

    machine_config: MachineConfig = make_machine_config(
        task=task,
        model_group_option=model_group_option,
        experiment_stage=experiment_stage,
        experiment_selector=experiment_selector,
        template_to_use_for_compute_embeddings=template_to_use_for_compute_embeddings,
        embedding_data_handler_mode=embedding_data_handler_mode,
        memory=memory,
        ncpus=ncpus,
        ngpus=ngpus,
        queue=queue,
        template=template,
    )

    if data_subsampling_sampling_mode == DataSamplingMode.TAKE_FIRST:
        # We do not need sampling seeds for the take first subsampling mode
        data_subsampling_sampling_seed_list_option = DataSubsamplingSamplingSeedListOption.NONE

    additional_overrides_parameter: list[str] | None = list(additional_overrides) if additional_overrides else None
    print(  # noqa: T201 - We want this script to print this output
        f"{additional_overrides_parameter = }",
    )

    make_submission_config_and_run_task(
        data_list_option=data_list_option,
        data_subsampling_sampling_mode=data_subsampling_sampling_mode,
        data_subsampling_number_of_samples_list_option=data_subsampling_number_of_samples_list_option,
        data_subsampling_sampling_seed_list_option=data_subsampling_sampling_seed_list_option,
        embeddings_data_prep_sampling_mode=embeddings_data_prep_sampling_mode,
        embeddings_data_prep_sampling_seed_list_option=embeddings_data_prep_sampling_seed_list_option,
        embeddings_data_prep_num_samples_list_option=embeddings_data_prep_num_samples_list_option,
        finetuning_base_model_list_option=finetuning_base_model_list_option,
        finetuning_datasets_list_option=finetuning_datasets_list_option,
        finetuning_seed_list_option=finetuning_seed_list_option,
        finetuning_regime_option=finetuning_regime_option,
        fp16=fp16,
        batch_size_train=batch_size_train,
        batch_size_eval=batch_size_eval,
        wandb_project=wandb_project,
        embedding_data_handler_mode=embedding_data_handler_mode,
        language_model_list_option=language_model_list_option,
        language_model_seed_list_option=language_model_seed_list_option,
        checkpoint_no_list_option=checkpoint_no_list_option,
        layer_indices_list=layer_indices_list,
        local_estimates_filtering_num_samples_list_option=local_estimates_filtering_num_samples_list_option,
        local_estimates_pointwise_absolute_n_neighbors_list_option=local_estimates_pointwise_absolute_n_neighbors_list_option,
        machine_config=machine_config,
        submission_mode=submission_mode,
        task=task,
        additional_overrides=additional_overrides_parameter,
        add_prefix_space=add_prefix_space,
        create_pos_tags=create_pos_tags,
        skip_compute_and_store_embeddings_in_pipeline=skip_compute_and_store_embeddings_in_pipeline,
        skip_embeddings_data_prep_in_pipeline=skip_embeddings_data_prep_in_pipeline,
        run_option=run_option,
        run_only_selected_configs_option=run_only_selected_configs_option,
    )


def main() -> None:
    """Run the job submission."""
    orchestrate_job_submission()


if __name__ == "__main__":
    orchestrate_job_submission()

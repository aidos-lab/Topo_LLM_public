#!/bin/bash

# DO_PIPELINE="true"
# DO_PERPLEXITY="true"
# DO_LOCAL_ESTIMATES_COMPUTATION="false"

DO_PIPELINE="false"
DO_PERPLEXITY="false"
DO_LOCAL_ESTIMATES_COMPUTATION="true"

# ================================================================== #
# Debug setup

# DATA_LIST="only_train"
# LANGUAGE_MODEL_LIST="selected_finetuned_few_epochs_from_roberta_base"
# CHECKPOINT_NO_LIST="selected"

# DATA_LIST="debug"
# LANGUAGE_MODEL_LIST="only_roberta_base"

# ================================================================== #

# EMBEDDINGS_DATA_PREP_SAMPLING_MODE="random"
EMBEDDINGS_DATA_PREP_SAMPLING_MODE="take_first"

# LOCAL_ESTIMATES_FILTERING_NUM_SAMPLES_LIST="few_small_steps_num_samples"
LOCAL_ESTIMATES_FILTERING_NUM_SAMPLES_LIST="medium_small_steps_num_samples"
# LOCAL_ESTIMATES_FILTERING_NUM_SAMPLES_LIST="many_small_steps_num_samples"
# LOCAL_ESTIMATES_FILTERING_NUM_SAMPLES_LIST="many_large_steps_num_samples"

### Without POS tags for multiwoz21_and_reddit but many checkpoints
#
# DATA_LIST="multiwoz21_and_reddit"
#
# LANGUAGE_MODEL_LIST="selected_finetuned_many_epochs_from_roberta_base"
# CHECKPOINT_NO_LIST="selected"
# LANGUAGE_MODEL_SEED_LIST="do_not_set"
# FINETUNING_REGIME="many_epochs_with_overfitting_risk"
# ADD_PREFIX_SPACE_FLAG=""
# CREATE_POS_TAGS_FLAG=""

### With POS tags for finetuned models and three checkpoints and two seeds
#
# DATA_LIST="full"
DATA_LIST="multiwoz21_and_reddit"

LANGUAGE_MODEL_LIST="selected_finetuned_many_epochs_from_roberta_base"
LANGUAGE_MODEL_SEED_LIST="one_seed"
CHECKPOINT_NO_LIST="only_beginning_and_middle_and_end"
FINETUNING_REGIME="many_epochs_with_overfitting_risk"
ADD_PREFIX_SPACE_FLAG="--add_prefix_space"
CREATE_POS_TAGS_FLAG="--create_pos_tags"

### With POS tags for base model
#
# DATA_LIST="full"
# DATA_LIST="multiwoz21_and_reddit"

# LANGUAGE_MODEL_LIST="only_roberta_base"
# LANGUAGE_MODEL_SEED_LIST="do_not_set"
# CHECKPOINT_NO_LIST="selected" # Will be ignored for the base model
# FINETUNING_REGIME="few_epochs" # Will be ignored for the base model
# ADD_PREFIX_SPACE_FLAG="--add_prefix_space"
# CREATE_POS_TAGS_FLAG="--create_pos_tags"

# ================================================================== #

SUBMISSION_MODE="hpc_submission"
# SUBMISSION_MODE="local"

# DRY_RUN_FLAG="--dry_run"

# ================================================================== #

if [ "$DO_PIPELINE" = "true" ]; then
    # Note: To avoid re-running the expensive steps
    # - compute_embeddings and
    # - embeddings_data_prep,
    # we do not use the 'LocalEstimatesFilteringNumSamplesOption' in the pipeline.
    poetry run submit_jobs \
        --task="pipeline" \
        --queue="DSML" \
        --template="DSML" \
        --data_list=$DATA_LIST \
        $CREATE_POS_TAGS_FLAG \
        --language_model_list=$LANGUAGE_MODEL_LIST \
        --checkpoint_no_list=$CHECKPOINT_NO_LIST \
        --language_model_seed_list=$LANGUAGE_MODEL_SEED_LIST \
        $ADD_PREFIX_SPACE_FLAG \
        --finetuning_regime=$FINETUNING_REGIME \
        --embeddings_data_prep_sampling_mode=$EMBEDDINGS_DATA_PREP_SAMPLING_MODE \
        --submission_mode=$SUBMISSION_MODE \
        $DRY_RUN_FLAG
fi

if [ "$DO_PERPLEXITY" = "true" ]; then
    # Note: Do not use the CREATE_POS_TAGS_FLAG for the perplexity task.
    poetry run submit_jobs \
        --task="perplexity" \
        --queue="CUDA" \
        --template="RTX6000" \
        --data_list=$DATA_LIST \
        --language_model_list=$LANGUAGE_MODEL_LIST \
        --checkpoint_no_list=$CHECKPOINT_NO_LIST \
        --language_model_seed_list=$LANGUAGE_MODEL_SEED_LIST \
        $ADD_PREFIX_SPACE_FLAG \
        --finetuning_regime=$FINETUNING_REGIME \
        --submission_mode=$SUBMISSION_MODE \
        $DRY_RUN_FLAG
fi

       

if [ "$DO_LOCAL_ESTIMATES_COMPUTATION" = "true" ]; then
    # This is a CPU task, so we do not ask for a GPU.
    poetry run submit_jobs \
        --task="local_estimates_computation" \
        --template="CPU" \
        --queue="DEFAULT" \
        --ngpus="0" \
        --walltime="10:00:00" \
        --data_list=$DATA_LIST \
        $CREATE_POS_TAGS_FLAG \
        --language_model_list=$LANGUAGE_MODEL_LIST \
        --checkpoint_no_list=$CHECKPOINT_NO_LIST \
        --language_model_seed_list=$LANGUAGE_MODEL_SEED_LIST \
        $ADD_PREFIX_SPACE_FLAG \
        --finetuning_regime=$FINETUNING_REGIME \
        --embeddings_data_prep_sampling_mode=$EMBEDDINGS_DATA_PREP_SAMPLING_MODE \
        --local_estimates_filtering_num_samples_list=$LOCAL_ESTIMATES_FILTERING_NUM_SAMPLES_LIST \
        --submission_mode=$SUBMISSION_MODE \
        $DRY_RUN_FLAG
fi

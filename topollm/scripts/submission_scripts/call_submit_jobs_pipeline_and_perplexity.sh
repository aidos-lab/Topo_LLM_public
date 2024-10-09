#!/bin/bash

# DATA_LIST="only_train"
# LANGUAGE_MODEL_LIST="selected_finetuned_few_epochs_from_roberta_base"
# CHECKPOINT_NO_LIST="selected"

# DATA_LIST="multiwoz21_and_reddit"
# LANGUAGE_MODEL_LIST="selected_finetuned_many_epochs_from_roberta_base"
# CHECKPOINT_NO_LIST="selected"
# LANGUAGE_MODEL_SEED_LIST="do_not_set"
# FINETUNING_REGIME="many_epochs_with_overfitting_risk"
# ADD_PREFIX_SPACE_FLAG=""

DATA_LIST="full"
LANGUAGE_MODEL_LIST="selected_finetuned_many_epochs_from_roberta_base"
CHECKPOINT_NO_LIST="selected"
LANGUAGE_MODEL_SEED_LIST="two_seeds"
FINETUNING_REGIME="many_epochs_with_overfitting_risk"
ADD_PREFIX_SPACE_FLAG="--add_prefix_space"

# DATA_LIST="debug"
# LANGUAGE_MODEL_LIST="only_roberta_base"

# ====================

# SUBMISSION_MODE="local"
SUBMISSION_MODE="hpc_submission"

DRY_RUN_FLAG="--dry_run"
# DRY_RUN_FLAG=""

poetry run submit_jobs \
    --task="pipeline" \
    --queue="DSML" \
    --template="DSML" \
    --data_list=$DATA_LIST \
    --language_model_list=$LANGUAGE_MODEL_LIST \
    --checkpoint_no_list=$CHECKPOINT_NO_LIST \
    --language_model_seed_list=$LANGUAGE_MODEL_SEED_LIST \
    $ADD_PREFIX_SPACE_FLAG \
    --finetuning_regime=$FINETUNING_REGIME \
    --submission_mode=$SUBMISSION_MODE \
    $DRY_RUN_FLAG

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

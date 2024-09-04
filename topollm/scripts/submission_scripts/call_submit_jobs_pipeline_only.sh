#!/bin/bash

DATA_LIST="multiwoz21_only"
LANGUAGE_MODEL_LIST="setsumbt_selected"

# Note: These are not necessary for this setup
CHECKPOINT_NO_LIST="selected"
FINETUNING_REGIME="few_epochs"

# ====================

SUBMISSION_MODE="local"
# SUBMISSION_MODE="hpc_submission"

# DRY_RUN_FLAG="--dry_run"
DRY_RUN_FLAG=""

poetry run submit_jobs \
    --task="pipeline" \
    --queue="DSML" \
    --template="DSML" \
    --data_list=$DATA_LIST \
    --language_model_list=$LANGUAGE_MODEL_LIST \
    --checkpoint_no_list=$CHECKPOINT_NO_LIST \
    --finetuning_regime=$FINETUNING_REGIME \
    --submission_mode=$SUBMISSION_MODE \
    $DRY_RUN_FLAG


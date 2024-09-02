#!/bin/bash

# DATA_LIST="only_train"
# LANGUAGE_MODEL_LIST="selected_finetuned_few_epochs_from_roberta_base"

DATA_LIST="debug"
LANGUAGE_MODEL_LIST="only_roberta_base"

# SUBMISSION_MODE="local"
SUBMISSION_MODE="hpc_submission"

poetry run submit_jobs \
    --task="pipeline" \
    --queue="DSML" \
    --template="DSML" \
    --data_list=$DATA_LIST \
    --language_model_list=$LANGUAGE_MODEL_LIST \
    --submission_mode=$SUBMISSION_MODE

poetry run submit_jobs \
    --task="perplexity" \
    --queue="CUDA" \
    --template="RTX6000" \
    --data_list=$DATA_LIST \
    --language_model_list=$LANGUAGE_MODEL_LIST \
    --submission_mode=$SUBMISSION_MODE

#!/bin/bash

# SUBMISSION_MODE="local"
SUBMISSION_MODE="hpc_submission"

poetry run submit_jobs \
    --task="finetuning" \
    --queue="CUDA" \
    --template="RTX6000" \
    --submission_mode=$SUBMISSION_MODE \
    --finetuning_datasets_list="manual_in_python_script" \
    --finetuning_regime="many_epochs_with_overfitting_risk"


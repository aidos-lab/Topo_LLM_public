#!/bin/bash

# ==================================================== #
POETRY_ENV_PATH=$(poetry env info --path)
echo "Using Poetry environment at $POETRY_ENV_PATH"

source "$POETRY_ENV_PATH/bin/activate"

export HYDRA_FULL_ERROR=1
# ==================================================== #

# Check for the dry run option
DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry run mode enabled."
fi


# # # # # # # # # # # # # # # # # # # # # # # # # # #
# START: Python script - Command line arguments

SUBMIT_PIPELINE_JOBS="run_multiple_models"

MACHINE_CONFIGURATION="rtx6000"
# MACHINE_CONFIGURATION="gtx1080ti"

# END: Python script - Command line arguments
# # # # # # # # # # # # # # # # # # # # # # # # # # #

# ==================================================== #

python3 submit_pipeline_jobs_with_hydra_config.py \
    submit_finetuning_jobs="${SUBMIT_FINETUNING_JOBS}" \
    machine_configuration="${MACHINE_CONFIGURATION}" \
    machine_configuration.dry_run="${DRY_RUN}"

# ==================================================== #

# Exit with the return code of the Python script.
exit $?
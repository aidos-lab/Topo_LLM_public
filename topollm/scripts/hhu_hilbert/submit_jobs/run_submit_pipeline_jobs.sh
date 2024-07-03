#!/bin/bash

# ==================================================== #
POETRY_ENV_PATH=$(poetry env info --path)
echo "Using Poetry environment at $POETRY_ENV_PATH"

source "$POETRY_ENV_PATH/bin/activate"

export HYDRA_FULL_ERROR=1
# ==================================================== #

# Source the argument parsing script
source ./parse_args.sh "$@"


# # # # # # # # # # # # # # # # # # # # # # # # # # #
# START: Python script - Command line arguments

# SUBMIT_PIPELINE_JOBS="run_multiple_models"
SUBMIT_PIPELINE_JOBS="run_single_model_all_checkpoints"

# MACHINE_CONFIGURATION="rtx6000"
MACHINE_CONFIGURATION="gtx1080ti"

# END: Python script - Command line arguments
# # # # # # # # # # # # # # # # # # # # # # # # # # #

# ==================================================== #

# Echo the variables
echo "SUBMIT_PIPELINE_JOBS: ${SUBMIT_PIPELINE_JOBS}"
echo "MACHINE_CONFIGURATION: ${MACHINE_CONFIGURATION}"
echo "DRY_RUN: ${DRY_RUN}"
echo "JOB_RUN_MODE: ${JOB_RUN_MODE}"

python3 submit_pipeline_jobs_with_hydra_config.py \
    submit_jobs/submit_pipeline_jobs="${SUBMIT_PIPELINE_JOBS}" \
    submit_jobs/machine_configuration="${MACHINE_CONFIGURATION}" \
    submit_jobs.machine_configuration.dry_run="${DRY_RUN}" \
    ++submit_jobs.machine_configuration.job_run_mode="${JOB_RUN_MODE}"

# ==================================================== #

# Exit with the return code of the Python script.
exit $?
#!/bin/bash

POETRY_ENV_PATH=$(poetry env info --path)
echo "Using Poetry environment at $POETRY_ENV_PATH"

source "$POETRY_ENV_PATH/bin/activate"

# Check for the dry run option
DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry run mode enabled."
fi

# SUBMIT_FINETUNING_JOBS="run_combinations_for_single_r_choice"
SUBMIT_FINETUNING_JOBS="run_multiple_r_choices"

MACHINE_CONFIGURATION="rtx6000"
# MACHINE_CONFIGURATION="gtx1080ti"

python3 submit_finetuning_jobs_with_hydra_config.py \
    submit_finetuning_jobs="${SUBMIT_FINETUNING_JOBS}" \
    machine_configuration="${MACHINE_CONFIGURATION}" \
    machine_configuration.dry_run="${DRY_RUN}"

# Exit with the return code of the Python script.
exit $?
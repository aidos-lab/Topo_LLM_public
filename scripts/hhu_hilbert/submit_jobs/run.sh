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

python3 submit_jobs_with_hydra_config.py \
    submit_finetuning_jobs=run_combinations_for_single_r_choice \
    submit_finetuning_jobs.dry_run=$DRY_RUN

# Exit with the return code of the Python script.
exit $?
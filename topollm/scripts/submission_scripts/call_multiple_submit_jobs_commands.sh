#!/bin/bash

# ================================================================== #

# Default values
DRY_RUN_FLAG=""

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --dry_run)
      DRY_RUN_FLAG="--dry_run"
      shift # Remove --dry_run from processing
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# ================================================================== #

CALL_SUBMIT_JOBS_SHELL_SCRIPT_PATH=$TOPO_LLM_REPOSITORY_BASE_PATH/topollm/scripts/submission_scripts/call_submit_jobs_pipeline_and_perplexity_and_local_estimates.sh

echo ">>> CALL_SUBMIT_JOBS_SHELL_SCRIPT_PATH: $CALL_SUBMIT_JOBS_SHELL_SCRIPT_PATH"
echo ">>> DRY_RUN_FLAG: $DRY_RUN_FLAG"

TASK_FLAG="--do_pipeline"
# TASK_FLAG="--do_local_estimates_computation"

$CALL_SUBMIT_JOBS_SHELL_SCRIPT_PATH \
    $TASK_FLAG \
    --use_roberta_base_model \
    $DRY_RUN_FLAG

$CALL_SUBMIT_JOBS_SHELL_SCRIPT_PATH \
    $TASK_FLAG \
    --use_finetuned_model \
    $DRY_RUN_FLAG


# ================================================================== #

# Exit script
echo ">>> Script finished."
echo ">>> Exiting ..."
exit 0
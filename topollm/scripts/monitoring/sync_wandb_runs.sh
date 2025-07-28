#!/bin/bash


# Default values
DRY_RUN=false

WANDB_PROJECT_DIR_NAME=""
# WANDB_PROJECT_DIR_NAME="Topo_LLM_finetuning_from_submission_script_DEBUG_large_batch_size"
# WANDB_PROJECT_DIR_NAME="Topo_LLM_finetuning_from_submission_script_dropout_different_choices"
# WANDB_PROJECT_DIR_NAME="Topo_LLM_finetuning_from_submission_script_dropout_different_choices_with_parameter_and_gradient_logging"
# WANDB_PROJECT_DIR_NAME="Topo_LLM_finetuning_from_submission_script_for_5_epochs_and_linear_lr_schedule"
WANDB_PROJECT_DIR_NAME="Topo_LLM_roberta-base_finetuning_from_submission_script_for_5_epochs_and_linear_lr_schedule_freeze_lm_head"

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --dry-run)
      DRY_RUN=true
      shift # Remove --dry-run from processing
      ;;
    --wandb-project-dir-name)
      WANDB_PROJECT_DIR_NAME="$2"
      shift 2 # Remove --wandb-project-dir-name and its value
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# # # # # # # # # # # # # # # # # # # # # # # #
# Check if required variables are set
# Check if WANDB_PROJECT_DIR_NAME is provided
if [[ -z "${WANDB_PROJECT_DIR_NAME}" ]]; then
  echo "@@@ Error: --wandb-project-dir-name is required."
  echo "@@@ Exiting script now without doing anything."
  exit 1
else
  echo ">>> WANDB_PROJECT_DIR_NAME=${WANDB_PROJECT_DIR_NAME}"
fi

if [[ -z "${TOPO_LLM_REPOSITORY_BASE_PATH}" ]]; then
  echo "@@@ Error: TOPO_LLM_REPOSITORY_BASE_PATH is not set."
  echo "@@@ Exiting script now without doing anything."
  exit 1
else
  echo ">>> TOPO_LLM_REPOSITORY_BASE_PATH=${TOPO_LLM_REPOSITORY_BASE_PATH}"
fi

# Construct the full path to the project directory
WANDB_OUTPUT_DIR_PATH="${TOPO_LLM_REPOSITORY_BASE_PATH}/wandb_output_dir"
WANDB_PROJECT_DIR_PATH="${WANDB_OUTPUT_DIR_PATH}/${WANDB_PROJECT_DIR_NAME}/wandb"
echo ">>> WANDB_PROJECT_DIR_PATH=${WANDB_PROJECT_DIR_PATH}"

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Loop through all matching offline run directories.
# Note: No quotes around the wildcard pattern, because we want to expand it.
for OFFLINE_RUN_DIR in "${WANDB_PROJECT_DIR_PATH}"/offline-*; do
  # Check if any matches were found
  if [ -d "$OFFLINE_RUN_DIR" ]; then
    if [ "$DRY_RUN" = true ]; then
      echo ">>> [DRY RUN] Would sync: ${OFFLINE_RUN_DIR}"
    else
      echo ">>> Syncing ${OFFLINE_RUN_DIR} to Weights and Biases ..."
      
      uv run wandb sync \
        --include-offline \
        "${OFFLINE_RUN_DIR}"

      # Exit if the sync command fails for any reason
      if [ $? -ne 0 ]; then
          echo "@@@ Error: Sync failed for ${OFFLINE_RUN_DIR}"
          exit 1
      fi

      echo ">>> Syncing ${OFFLINE_RUN_DIR} to Weights and Biases DONE"
    fi
  fi
done

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Note: We usually do not need to sync the online runs, 
# because they are already synced automatically.

# Exit with the exit status of the last command
exit $?
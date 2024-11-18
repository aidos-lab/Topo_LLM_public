#!/bin/bash

DRY_RUN=false

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --dry_run)
      DRY_RUN=true
      shift # Remove --dry_run from processing
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# # # # # # # # # # # # # # # # # # # # # # # #
# Check if TOPO_LLM_REPOSITORY_BASE_PATH is set
if [[ -z "${TOPO_LLM_REPOSITORY_BASE_PATH}" ]]; then
  echo "@@@ Error: TOPO_LLM_REPOSITORY_BASE_PATH is not set."
  echo "@@@ Exiting script now without doing anything."
  exit 1
else
  echo ">>> TOPO_LLM_REPOSITORY_BASE_PATH=${TOPO_LLM_REPOSITORY_BASE_PATH}"
fi

WANDB_OUTPUT_DIR_PATH="${TOPO_LLM_REPOSITORY_BASE_PATH}/wandb_output_dir"
WANDB_PROJECT_DIR_NAME="Topo_LLM_finetuning_from_submission_script_DEBUG_large_batch_size"

# Construct the full path to the project directory
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
      
      wandb sync \
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
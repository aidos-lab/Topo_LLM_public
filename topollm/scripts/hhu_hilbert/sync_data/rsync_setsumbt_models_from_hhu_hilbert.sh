#!/bin/bash

# Function to print usage
usage() {
  echo "Usage: $0 [--dry-run]"
  exit 1
}

# Default value for dry_run option
DRY_RUN_FLAG=""

# Parse command-line options
if [[ $# -gt 1 ]]; then
  echo "Error: Too many arguments."
  usage
fi

if [[ $# -eq 1 ]]; then
  case "$1" in
    --dry-run)
      DRY_RUN_FLAG="--dry-run"
      ;;
    *)
      echo "Error: Invalid option $1"
      usage
      ;;
  esac
fi


# Check if TOPO_LLM_REPOSITORY_BASE_PATH is set
if [[ -z "${TOPO_LLM_REPOSITORY_BASE_PATH}" ]]; then
  echo "Error: TOPO_LLM_REPOSITORY_BASE_PATH is not set."
  exit 1
fi

source "${TOPO_LLM_REPOSITORY_BASE_PATH}/.env"

if [[ -z "${GPFS_PROJECT_DIR_DSML}" ]]; then
  echo "Error: GPFS_PROJECT_DIR_DSML is not set."
  exit 1
fi

# Print variables
echo "TOPO_LLM_REPOSITORY_BASE_PATH=$TOPO_LLM_REPOSITORY_BASE_PATH"
echo "GPFS_PROJECT_DIR_DSML=$GPFS_PROJECT_DIR_DSML"

# # # #
# NOTE: This script is only syncing a subset of the models from the HHU Hilbert server,
# which are not all checkpoints of the model. This is for space reasons on the local machine.

# Define the subfolder to sync
SELECTED_SUBFOLDER="multiwoz21/roberta/setsumbt/gru/cosine/labelsmoothing/0.05/seed0/"

SOURCE_DIR="Hilbert-Storage:${GPFS_PROJECT_DIR_DSML}/data/data-exp-eriments-zetsumbt/$SELECTED_SUBFOLDER"
# Create the target directory if it doesn't exist
TARGET_DIR="${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/setsumbt_checkpoints/$SELECTED_SUBFOLDER"
mkdir -p "$TARGET_DIR"

EXCLUDE_FROM_FILE="${TOPO_LLM_REPOSITORY_BASE_PATH}/topollm/scripts/hhu_hilbert/sync_data/rsync_setsumbt_models_excludes.txt"

echo "EXCLUDE_FROM_FILE=$EXCLUDE_FROM_FILE"

# ========================

echo "Syncing data from HHU Hilbert server to local machine ..."

# Following rsync instructions from:
# https://wiki.hhu.de/pages/viewpage.action?pageId=55725648
rsync \
  -avz \
  --progress \
  $DRY_RUN_FLAG \
  --exclude-from="${EXCLUDE_FROM_FILE}" \
  "${SOURCE_DIR}" \
  "${TARGET_DIR}"

# Capture the exit code of rsync
RSYNC_EXIT_CODE=$?
if [[ ${RSYNC_EXIT_CODE} -ne 0 ]]; then
  echo "Error: rsync failed with exit code ${RSYNC_EXIT_CODE}"
  exit ${RSYNC_EXIT_CODE}
fi

echo "rsync completed successfully."
exit 0
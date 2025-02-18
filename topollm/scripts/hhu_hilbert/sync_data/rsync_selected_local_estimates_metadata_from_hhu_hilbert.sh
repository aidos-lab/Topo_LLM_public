#!/bin/bash

# # # #
# NOTE: 
# This script syncs selected local estimates metadata from the HHU Hilbert server to the local machine.

# # # # # # # # # # # # # # # # # # # # # # # # #
# Default values
DRY_RUN_FLAG=""
REMOTE_HOST="Hilbert-Storage"
REMOTE_BASE_DIR="${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/data/analysis/local_estimates"

# TODO: This is not the correct pattern, since for example it matches all layers
PATH_TO_MATCH="*model=roberta-base-trippy_multiwoz21_seed-*_task=masked_lm_dr=defaults*local_estimates_pointwise_meta.pkl"
TARGET_DIR="${TOPO_LLM_REPOSITORY_BASE_PATH}/data/example_sync_from_found_files/"

# Function to print usage
usage() {
  echo ">>> Usage: $0 [--dry-run]"
  exit 1
}

# Parse command-line options
if [[ $# -gt 1 ]]; then
  echo "❌ Error: Too many arguments."
  usage
fi

if [[ $# -eq 1 ]]; then
  case "$1" in
    --dry-run)
      DRY_RUN_FLAG="--dry-run"
      ;;
    *)
      echo "❌ Error: Invalid option $1"
      usage
      ;;
  esac
fi

# # # # # # # # # # # # # # # # # # # # # # # # #
# Check if TOPO_LLM_REPOSITORY_BASE_PATH is set
if [[ -z "${TOPO_LLM_REPOSITORY_BASE_PATH}" ]]; then
  echo "❌ Error: TOPO_LLM_REPOSITORY_BASE_PATH is not set."
  exit 1
fi

source "${TOPO_LLM_REPOSITORY_BASE_PATH}/.env"

# # # # # # # # # # # # # # # # # # # # # # # # #
# Print variables
variables_to_log=(
  "TOPO_LLM_REPOSITORY_BASE_PATH"
  "ZIM_TOPO_LLM_REPOSITORY_BASE_PATH"
  "REMOTE_HOST"
  "REMOTE_BASE_DIR"
  "PATH_TO_MATCH"
  "TARGET_DIR"
)

for variable in "${variables_to_log[@]}"; do
  echo ">>> $variable=${!variable}"
done

# ======================================================================================== #
# Find the matching files on the remote machine
# ======================================================================================== #

# Find matching files on the remote machine
echo "🔍 Searching for matching files on remote machine ($REMOTE_HOST) ..."
MATCHED_FILES=$(ssh "$REMOTE_HOST" "find $REMOTE_BASE_DIR -type f -path '$PATH_TO_MATCH'")

# Check if any files were found
if [[ -z "$MATCHED_FILES" ]]; then
    echo "❌ No matching files found!"
    exit 1
fi

# Print matched files for debugging
MATCHED_FILE_COUNT=$(echo "$MATCHED_FILES" | wc -l)
echo "✅ Total matched files: $MATCHED_FILE_COUNT"
echo "✅ Matched files:"
echo "$MATCHED_FILES"

# ======================================================================================== #
# Run rsync to sync the matched files to the local machine
# ======================================================================================== #

# Create the target directory if it does not exist
mkdir -p "$TARGET_DIR"

# Perform the actual rsync
# Following rsync instructions from:
# https://wiki.hhu.de/pages/viewpage.action?pageId=55725648
echo "🚀 Starting file sync..."
echo "$MATCHED_FILES" | rsync -avhz --progress $DRY_RUN_FLAG --files-from=- "$REMOTE_HOST:/" "$TARGET_DIR/"

# TODO: This rsync command does not place the files at the correct locations in the local folder structure

# Capture the exit code of rsync
RSYNC_EXIT_CODE=$?
if [[ ${RSYNC_EXIT_CODE} -ne 0 ]]; then
  echo "❌ Error: rsync failed with exit code ${RSYNC_EXIT_CODE}"
  exit ${RSYNC_EXIT_CODE}
fi

# ======================================================================================== #

echo "✅ rsync completed successfully."
exit 0
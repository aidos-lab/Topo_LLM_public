#!/bin/bash

# Check if TOPO_LLM_REPOSITORY_BASE_PATH is set
if [[ -z "${TOPO_LLM_REPOSITORY_BASE_PATH}" ]]; then
  echo "Error: TOPO_LLM_REPOSITORY_BASE_PATH is not set."
  exit 1
fi

source "${TOPO_LLM_REPOSITORY_BASE_PATH}/.env"

# Print variables
echo "TOPO_LLM_REPOSITORY_BASE_PATH=$TOPO_LLM_REPOSITORY_BASE_PATH"
echo "ZIM_TOPO_LLM_REPOSITORY_BASE_PATH=$ZIM_TOPO_LLM_REPOSITORY_BASE_PATH"

# # # #
# NOTE: This script is only syncing a subset of the models from the HHU Hilbert server,
# which are not all checkpoints of the model. This is for space reasons on the local machine.

# Define the subfolder to sync
SELECTED_SUBFOLDER="multiwoz21/roberta/setsumbt/gru/cosine/labelsmoothing/0.05/seed0/"

# Create the target directory if it doesn't exist
TARGET_DIR="${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/setsumbt_checkpoints/$SELECTED_SUBFOLDER"
mkdir -p "$TARGET_DIR"

# Following rsync instructions from:
# https://wiki.hhu.de/pages/viewpage.action?pageId=55725648
rsync -avz --progress \
    "Hilbert-Storage:/gpfs/project/projects/dsml/data/data-exp-eriments-zetsumbt/$SELECTED_SUBFOLDER" \
    "$TARGET_DIR"

# Capture the exit code of rsync
RSYNC_EXIT_CODE=$?
if [[ ${RSYNC_EXIT_CODE} -ne 0 ]]; then
  echo "Error: rsync failed with exit code ${RSYNC_EXIT_CODE}"
  exit ${RSYNC_EXIT_CODE}
fi

echo "rsync completed successfully."
exit 0
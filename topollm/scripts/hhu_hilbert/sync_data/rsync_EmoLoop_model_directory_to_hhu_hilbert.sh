#!/bin/bash

# Following rsync instructions from:
# https://wiki.hhu.de/pages/viewpage.action?pageId=55725648

# # # # # # # # # # # # # # # # # # # # # # # # #
# Default values
DRY_RUN_FLAG=""

# # # # # # # # # # # # # # # # # # # # # # # # #
# Function to print usage
usage() {
    echo "💡 Usage: $0 [--dry-run]"
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

# Load environment variables
source "${TOPO_LLM_REPOSITORY_BASE_PATH}/.env"

echo "TOPO_LLM_REPOSITORY_BASE_PATH=$TOPO_LLM_REPOSITORY_BASE_PATH"

# Check if the ZIM_USERNAME variable is set
if [ -z "$ZIM_USERNAME" ]; then
    echo "Error: ZIM_USERNAME is not set."
    exit 1
fi

# # # # # # # # # # # # # # # # # # # # # # # # #
# Sync command

SOURCE_PATH="${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/EmoLoop/"
TARGET_PATH="${ZIM_USERNAME}@${REMOTE_HOST}:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/EmoLoop/"

echo "Syncing from ${SOURCE_PATH} to ${TARGET_PATH} ..."

rsync -avhz --delete --progress $DRY_RUN_FLAG \
    "${SOURCE_PATH}" \
    "${TARGET_PATH}"

echo "Sync completed."

# Capture the exit code of rsync
RSYNC_EXIT_CODE=$?
if [[ ${RSYNC_EXIT_CODE} -ne 0 ]]; then
    echo "❌ Error: rsync failed with exit code ${RSYNC_EXIT_CODE}"
    exit ${RSYNC_EXIT_CODE}
fi

# ======================================================================================== #

echo "✅ rsync completed successfully."
exit 0

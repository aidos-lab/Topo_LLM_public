#!/bin/bash

# # # # # # # # # # # # # # # # # # # # # # # # #
# Default values
DRY_RUN_FLAG=""
REMOTE_HOST="Hilbert-Storage"

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
# Check if environment variables are set

if [[ -z "${TOPO_LLM_REPOSITORY_BASE_PATH}" ]]; then
  echo "❌ Error: TOPO_LLM_REPOSITORY_BASE_PATH is not set."
  exit 1
fi

source "${TOPO_LLM_REPOSITORY_BASE_PATH}/.env"

if [[ -z "${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}" ]]; then
  echo "❌ Error: ZIM_TOPO_LLM_REPOSITORY_BASE_PATH is not set."
  exit 1
fi

# # # # # # # # # # # # # # # # # # # # # # # # #
# Log variables

VARIABLES_TO_LOG_LIST=(
    "TOPO_LLM_REPOSITORY_BASE_PATH"
    "ZIM_TOPO_LLM_REPOSITORY_BASE_PATH"
)

for VARIABLE_NAME in "${VARIABLES_TO_LOG_LIST[@]}"; do
    echo "💡 ${VARIABLE_NAME}=${!VARIABLE_NAME}"
done

# # # # # # # # # # # # # # # # # # # # # # # # #
# Sync data using rsync

echo ">>> Starting rsync ..."

# Following rsync instructions from:
# https://wiki.hhu.de/pages/viewpage.action?pageId=55725648
rsync \
    -avhz \
    --progress \
    ${DRY_RUN_FLAG} \
    "${REMOTE_HOST}:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_checkpoints/" \
    "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_checkpoints/"

echo ">>> rsync completed."

# Exit with the exit code of the rsync command
exit $?
#!/bin/bash

# Following rsync instructions from:
# https://wiki.hhu.de/pages/viewpage.action?pageId=55725648

echo "TOPO_LLM_REPOSITORY_BASE_PATH=$TOPO_LLM_REPOSITORY_BASE_PATH"

source "${TOPO_LLM_REPOSITORY_BASE_PATH}/.env"

# Check if the LOCAL_HUGGINGFACE_CACHE_PATH variable is set
if [ -z "$LOCAL_HUGGINGFACE_CACHE_PATH" ]; then
    echo "Error: LOCAL_HUGGINGFACE_CACHE_PATH is not set."
    exit 1
fi
# Check if the ZIM_USERNAME variable is set
if [ -z "$ZIM_USERNAME" ]; then
    echo "Error: ZIM_USERNAME is not set."
    exit 1
fi

SOURCE_PATH="${LOCAL_HUGGINGFACE_CACHE_PATH}"
TARGET_PATH="${ZIM_USERNAME}@Hilbert-Storage:/gpfs/project/${ZIM_USERNAME}/models/"

echo "Syncing from ${SOURCE_PATH} to ${TARGET_PATH} ..."

rsync -avhz --delete --progress \
    "${SOURCE_PATH}" \
    "${TARGET_PATH}"

echo "Sync completed."

# Exit with the exit code of the rsync command
exit $?
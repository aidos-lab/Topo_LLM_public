#!/bin/bash

echo "TOPO_LLM_REPOSITORY_BASE_PATH=$TOPO_LLM_REPOSITORY_BASE_PATH"

source "${TOPO_LLM_REPOSITORY_BASE_PATH}/.env"

# # # # # # # # # # # # # # # # # # # # # # # # #
# Check if variables are set
check_variable() {
    local var_name="$1"
    local var_value="${!var_name}"
    if [[ -z "${var_value}" ]]; then
        echo "❌ Error: ${var_name} is not set."
        exit 1
    else
        echo "✅ ${var_name}=${var_value}"
    fi
}

check_variable "TOPO_LLM_REPOSITORY_BASE_PATH"
check_variable "LOCAL_HUGGINGFACE_CACHE_PATH"
check_variable "REMOTE_HOST"
check_variable "ZIM_TOPO_LLM_REPOSITORY_BASE_PATH"
check_variable "ZIM_USERNAME"

SOURCE_PATH="${LOCAL_HUGGINGFACE_CACHE_PATH}"
TARGET_PATH="${ZIM_USERNAME}@${REMOTE_HOST}:/gpfs/project/${ZIM_USERNAME}/models/"

echo "Syncing from ${SOURCE_PATH} to ${TARGET_PATH} ..."

# Following rsync instructions from:
# https://wiki.hhu.de/pages/viewpage.action?pageId=55725648
rsync \
    -avhz --delete --progress \
    "${SOURCE_PATH}" \
    "${TARGET_PATH}"

echo "Sync completed."

# Exit with the exit code of the rsync command
exit $?

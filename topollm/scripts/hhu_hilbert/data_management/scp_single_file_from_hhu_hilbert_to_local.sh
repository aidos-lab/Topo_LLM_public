#!/bin/bash

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
check_variable "REMOTE_HOST"
check_variable "ZIM_USERNAME"

echo ">>> Copying file from ${REMOTE_HOST} to local machine..."

scp "${REMOTE_HOST}:/gpfs/project/${ZIM_USERNAME}/git-source/Topo_LLM/data/models/setsumbt_checkpoints/rsync_output.txt" .

if [ $? -eq 0 ]; then
    echo ">>> File copied successfully."
else
    echo ">>> Error copying file."
fi

exit 0

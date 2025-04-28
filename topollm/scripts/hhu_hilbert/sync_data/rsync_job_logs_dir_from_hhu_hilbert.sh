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

# Following rsync instructions from:
# https://wiki.hhu.de/pages/viewpage.action?pageId=55725648
rsync \
    -avhz --progress \
    "${REMOTE_HOST}:/gpfs/project/${ZIM_USERNAME}/job_logs/" \
    "${TOPO_LLM_REPOSITORY_BASE_PATH}/hilbert_job_logs/"

# Exit with the exit code of the rsync command
exit $?

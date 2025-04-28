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

LOCAL_NLTK_DATA_PATH="$HOME/nltk_data/"

# Following rsync instructions from:
# https://wiki.hhu.de/pages/viewpage.action?pageId=55725648
rsync \
    -avhz --delete --progress \
    "${LOCAL_NLTK_DATA_PATH}" \
    "${ZIM_USERNAME}@${REMOTE_HOST}:/home/${ZIM_USERNAME}/nltk_data/"

# Exit with the exit code of the rsync command
exit $?

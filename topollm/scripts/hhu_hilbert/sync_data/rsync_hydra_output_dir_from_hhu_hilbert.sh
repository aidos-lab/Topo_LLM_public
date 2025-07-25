#!/bin/bash

SUBDIRECTORIES_TO_SYNC=(
    "multirun"
    # "run"
)

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
check_variable "ZIM_TOPO_LLM_REPOSITORY_BASE_PATH"
check_variable "REMOTE_HOST"

for subdir in "${SUBDIRECTORIES_TO_SYNC[@]}"; do
    SOURCE="${REMOTE_HOST}:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/hydra_output_dir/${subdir}/"
    DESTINATION="${TOPO_LLM_REPOSITORY_BASE_PATH}/hydra_output_dir/${subdir}/"

    # Following rsync instructions from:
    # https://wiki.hhu.de/pages/viewpage.action?pageId=55725648
    rsync \
        -avhz --progress \
        "${SOURCE}" \
        "${DESTINATION}"
done



# Exit with the exit code of the rsync command
exit $?

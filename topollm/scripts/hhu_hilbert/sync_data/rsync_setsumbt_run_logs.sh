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
check_variable "ZIM_TOPO_LLM_REPOSITORY_BASE_PATH"

SELECTED_FILE_PATH_LIST=(
    "data/models/setsumbt_checkpoints/multiwoz21/roberta/setsumbt/gru/cosine/labelsmoothing/0.05/seed0/latest-run/run.log"
    "data/models/setsumbt_checkpoints/multiwoz21/roberta/setsumbt/gru/cosine/labelsmoothing/0.05/seed0/latest-run/run.jsonl"
    "data/models/setsumbt_checkpoints/multiwoz21/roberta/setsumbt/gru/cosine/labelsmoothing/0.05/seed1/latest-run/run.log"
    "data/models/setsumbt_checkpoints/multiwoz21/roberta/setsumbt/gru/cosine/labelsmoothing/0.05/seed1/latest-run/run.jsonl"
    "data/models/setsumbt_checkpoints/multiwoz21/roberta/setsumbt/gru/cosine/labelsmoothing/0.05/seed2/latest-run/run.log"
    "data/models/setsumbt_checkpoints/multiwoz21/roberta/setsumbt/gru/cosine/labelsmoothing/0.05/seed2/latest-run/run.jsonl"
)

for SELECTED_FILE_PATH in "${SELECTED_FILE_PATH_LIST[@]}"; do
    SOURCE_FILE_PATH="${REMOTE_HOST}:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/$SELECTED_FILE_PATH"
    TARGET_FILE_PATH="${TOPO_LLM_REPOSITORY_BASE_PATH}/$SELECTED_FILE_PATH"

    echo ">>> SELECTED_FILE_PATH=$SELECTED_FILE_PATH"
    echo ">>> SOURCE_FILE_PATH=$SOURCE_FILE_PATH"
    echo ">>> TARGET_FILE_PATH=$TARGET_FILE_PATH"

    # Create the target directory if it does not exist
    mkdir -p "$(dirname $TARGET_FILE_PATH)"

    # ========================
    echo ">>> Syncing data from HHU Hilbert server to local machine ..."

    rsync \
        -avhz \
        --progress \
        "${SOURCE_FILE_PATH}" \
        "${TARGET_FILE_PATH}"

done

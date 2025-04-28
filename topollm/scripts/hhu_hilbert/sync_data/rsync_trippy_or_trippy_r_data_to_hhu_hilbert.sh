#!/bin/bash

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
# Check if environment variables are set

if [[ -z "${TOPO_LLM_REPOSITORY_BASE_PATH}" ]]; then
    echo "❌ Error: TOPO_LLM_REPOSITORY_BASE_PATH is not set."
    exit 1
fi

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
check_variable "ZIM_TOPO_LLM_REPOSITORY_BASE_PATH"

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
# Define sync pairs using two parallel arrays.
#
# - Each source in SYNC_SRC corresponds to the destination at the same index in SYNC_DEST.
# - Comment out any pair you do not want to sync.
# - Remember the slash at the end of the source path and the destination path.

SYNC_SRC=(
    "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_checkpoints/post_processed_cached_features/"
    "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/post_processed_cached_features/"
)

SYNC_DEST=(
    "${REMOTE_HOST}:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_checkpoints/post_processed_cached_features/"
    "${REMOTE_HOST}:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/post_processed_cached_features/"
)

# Ensure both arrays have the same number of active entries.
if [[ ${#SYNC_SRC[@]} -ne ${#SYNC_DEST[@]} ]]; then
    echo "❌ Error: The number of source paths and destination paths do not match."
    exit 1
fi

echo ">>> Starting rsync operations ..."

overall_exit=0

for i in "${!SYNC_SRC[@]}"; do
    SRC="${SYNC_SRC[i]}"
    DEST="${SYNC_DEST[i]}"

    echo "===================================================="
    echo "🔄 Syncing: '${SRC}' -> '${DEST}'"

    # Create the destination directory if it does not exist
    mkdir -p "$DEST"

    # Following rsync instructions from:
    # https://wiki.hhu.de/pages/viewpage.action?pageId=55725648
    rsync \
        -avhz \
        --progress \
        ${DRY_RUN_FLAG} \
        "$SRC" \
        "$DEST"
    rc=$?

    if [[ $rc -ne 0 ]]; then
        echo "❌ Error: rsync failed for '${SRC}' -> '${DEST}'"
        overall_exit=$rc
    fi

    echo "===================================================="
done

echo ">>> All rsync operations completed."
exit $overall_exit

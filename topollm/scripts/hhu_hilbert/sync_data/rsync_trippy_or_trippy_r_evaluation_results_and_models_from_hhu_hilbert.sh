#!/bin/bash

# # # # # # # # # # # # # # # # # # # # # # # # #
# Default values
DRY_RUN_FLAG=""
REMOTE_HOST="HilbertStorage"

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
# Define sync pairs using two parallel arrays.
#
# - Each source in SYNC_SRC corresponds to the destination at the same index in SYNC_DEST.
# - Comment out any pair you do not want to sync.
# - Remember the slash at the end of the source path and the destination path.

SYNC_SRC=(
    # "${REMOTE_HOST}:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_checkpoints/"
    # "${REMOTE_HOST}:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints_code/"
    # "${REMOTE_HOST}:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/"
    # "${REMOTE_HOST}:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/not_domain_code/"
    # "${REMOTE_HOST}:/gpfs/project/projects/dsml/data/multiwoz21/all_checkpoints_code/"
    # "${REMOTE_HOST}:/gpfs/project/projects/dsml/data/multiwoz21/all_checkpoints/results.42/"
    # "${REMOTE_HOST}:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/cached_dev_features"
    # "${REMOTE_HOST}:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/cached_test_features"
)

# Extend the SYNC_SRC array with additional files for each seed
SEEDS=(
    1111
    40
    41
    42
    43
    44
)

for SEED in "${SEEDS[@]}"; do
    SYNC_SRC+=(
        "${REMOTE_HOST}:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.${SEED}/dev.-1.log"
        "${REMOTE_HOST}:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.${SEED}/test.-1.log"
        "${REMOTE_HOST}:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.${SEED}/train.-1.log"
        "${REMOTE_HOST}:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.${SEED}/eval_res.dev.json"
        "${REMOTE_HOST}:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.${SEED}/eval_res.test.json"
        "${REMOTE_HOST}:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.${SEED}/eval_res.train.json"
        "${REMOTE_HOST}:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.${SEED}/eval_pred_dev.1.0.log"
        "${REMOTE_HOST}:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.${SEED}/eval_pred_test.1.0.log"
        "${REMOTE_HOST}:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.${SEED}/eval_pred_train.1.0.log"
    )
done

SYNC_DEST=(
    # "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_checkpoints/"
    # "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints_code/"
    # "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/"
    # "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/not_domain_code/"
    # "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints_code/"
    # "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.42_downloaded_from_michael/"
    # "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/" # No target file name here, to place the files in the same directory
    # "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/" # No target file name here, to place the files in the same directory
)

# Extend the SYNC_DEST array with additional files for each seed
for SEED in "${SEEDS[@]}"; do
    # Notes:
    # - No target file names here, to place the files in the same directory
    # - There should be the same number of source and destination paths
    SYNC_DEST+=(
        "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.${SEED}/"
        "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.${SEED}/"
        "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.${SEED}/"
        "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.${SEED}/"
        "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.${SEED}/"
        "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.${SEED}/"
        "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.${SEED}/"
        "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.${SEED}/"
        "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.${SEED}/"
    )
done

# Ensure both arrays have the same number of active entries.
if [[ ${#SYNC_SRC[@]} -ne ${#SYNC_DEST[@]} ]]; then
    echo "❌ Error: The number of source paths and destination paths do not match."
    exit 1
fi

echo ">>> Starting rsync operations ..."

# Optional: Add a delay between operations to avoid overwhelming the server
DELAY_BETWEEN_OPERATIONS=1
DELAY_IF_ERROR=3

# Initialize the single operation return code to 0
rc=0
overall_exit=0

for i in "${!SYNC_SRC[@]}"; do
    SRC="${SYNC_SRC[i]}"
    DEST="${SYNC_DEST[i]}"

    # Optional: Add a delay between operations
    echo "💡 Adding a delay of ${DELAY_BETWEEN_OPERATIONS} seconds before the next operation ..."
    sleep $DELAY_BETWEEN_OPERATIONS

    # If the last operation failed, add a longer delay
    if [[ $rc -ne 0 ]]; then
        echo "💡 Last operation failed. Adding a longer delay of ${DELAY_IF_ERROR} seconds ..."
        sleep $DELAY_IF_ERROR
    fi

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

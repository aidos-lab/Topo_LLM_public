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
    # # # > seed=42
    "${REMOTE_HOST}:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.42/eval_res.dev.json"
    "${REMOTE_HOST}:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.42/eval_res.test.json"
    "${REMOTE_HOST}:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.42/eval_pred_dev.1.0.log"
    "${REMOTE_HOST}:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.42/eval_pred_test.1.0.log"
    # # # > seed=43
    "${REMOTE_HOST}:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.43/eval_res.dev.json"
    "${REMOTE_HOST}:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.43/eval_res.test.json"
    "${REMOTE_HOST}:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.43/eval_pred_dev.1.0.log"
    "${REMOTE_HOST}:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.43/eval_pred_test.1.0.log"
    # # # > seed=44
    "${REMOTE_HOST}:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.44/eval_res.dev.json"
    "${REMOTE_HOST}:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.44/eval_res.test.json"
    "${REMOTE_HOST}:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.44/eval_pred_dev.1.0.log"
    "${REMOTE_HOST}:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.44/eval_pred_test.1.0.log"
)

SYNC_DEST=(
    # "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_checkpoints/"
    # "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints_code/"
    # "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/"
    # "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/not_domain_code/"
    # "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints_code/"
    # "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.42_downloaded_from_michael/"
    # "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/" # No target file name here, to place the files in the same directory
    # "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/" # No target file name here, to place the files in the same directory
    # # # > seed=42
    "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.42/"
    "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.42/"
    "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.42/"
    "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.42/"
    # # # > seed=43
    "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.43/"
    "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.43/"
    "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.43/"
    "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.43/"
    # # # > seed=44
    "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.44/"
    "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.44/"
    "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.44/"
    "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.44/"
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

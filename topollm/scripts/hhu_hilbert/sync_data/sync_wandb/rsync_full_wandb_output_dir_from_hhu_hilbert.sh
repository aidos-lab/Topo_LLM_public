#!/bin/bash

# Function to print usage
usage() {
    echo ">>> Usage: $0 [--dry-run]"
    exit 1
}

# Default value for dry_run option
DRY_RUN_FLAG=""

# Parse command-line options
if [[ $# -gt 1 ]]; then
    echo ">>> Error: Too many arguments."
    usage
fi

if [[ $# -eq 1 ]]; then
    case "$1" in
    --dry-run)
        DRY_RUN_FLAG="--dry-run"
        ;;
    *)
        echo ">>> Error: Invalid option $1"
        usage
        ;;
    esac
fi

# # # # # # # # # # # # # # # # # # # # # # # # #
# Check if TOPO_LLM_REPOSITORY_BASE_PATH is set
if [[ -z "${TOPO_LLM_REPOSITORY_BASE_PATH}" ]]; then
    echo ">>> Error: TOPO_LLM_REPOSITORY_BASE_PATH is not set."
    exit 1
fi

source "${TOPO_LLM_REPOSITORY_BASE_PATH}/.env"

# # # # # # # # # # # # # # # # # # # # # # # # #
# Print variables
echo ">>> TOPO_LLM_REPOSITORY_BASE_PATH=$TOPO_LLM_REPOSITORY_BASE_PATH"

REPOSITORY_SUBDIRECTORY_PATH_LIST=(
    "wandb_output_dir/"
)

SCRIPT_PATH="${TOPO_LLM_REPOSITORY_BASE_PATH}/topollm/scripts/cluster_backend/sync_data/rsync_selected_directories_from_cluster_backend.sh"

${SCRIPT_PATH} \
    ${DRY_RUN_FLAG} \
    --folders "${REPOSITORY_SUBDIRECTORY_PATH_LIST[@]}"

exit 0

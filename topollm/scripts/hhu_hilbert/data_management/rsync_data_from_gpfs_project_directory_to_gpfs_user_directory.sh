#!/bin/bash

# # # # # # # # # # # # # # # # # # # # # # # # #
# This script syncs data from the gpfs project directory to the gpfs user directory using rsync.
#
# Notes:
# - This script should be run of the storage node on the cluster.

set -e  # Exit immediately if a command exits with a non-zero status.

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

# Load environment variables from the .env file
source "${TOPO_LLM_REPOSITORY_BASE_PATH}/.env"

if [[ -z "${GPFS_PROJECT_DIR_DSML}" ]]; then
    echo "❌ Error: GPFS_PROJECT_DIR_DSML is not set."
    exit 1
fi

# # # # # # # # # # # # # # # # # # # # # # # # #
# Log variables

VARIABLES_TO_LOG_LIST=(
    "TOPO_LLM_REPOSITORY_BASE_PATH" # example: /gpfs/project/$USER/git-source/Topo_LLM
    "GPFS_PROJECT_DIR_DSML" # > example: /gpfs/project/projects/dsml
)

for VARIABLE_NAME in "${VARIABLES_TO_LOG_LIST[@]}"; do
    echo "💡 ${VARIABLE_NAME}=${!VARIABLE_NAME}"
done

# # # # # # # # # # # # # # # # # # # # # # # # #
# Sync data using rsync

# Define an array of source:destination pairs.
#
# Notes:
# - The trailing slashes are important for rsync to copy the contents of the source directory into the destination.

pairs=(
    # > Example directory for the Trippy-R checkpoints:
    # > /gpfs/project/projects/dsml/data/trippy_r
    #
    # "${GPFS_PROJECT_DIR_DSML}/data/trippy_r/:${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/"
    # "${GPFS_PROJECT_DIR_DSML}/data/multiwoz21/:${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/"
    #
    # Example: Rsync from the snapshots of the project directory to the user directory
    "${GPFS_PROJECT_DIR_DSML}/.snapshots/day-2025.03.20-23.01.02/data/multiwoz21/:${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/"
)

# Iterate over each pair and run rsync for syncing the directories.
for pair in "${pairs[@]}"; do
    # Split the pair using ':' as the delimiter.
    IFS=":" read -r src dst <<< "$pair"
    

    echo ">>> Syncing data from ${src} to ${dst} ..."

    # Following rsync instructions from:
    # https://wiki.hhu.de/pages/viewpage.action?pageId=55725648
    rsync \
        -avhz \
        --progress \
        ${DRY_RUN_FLAG} \
        "${src}" \
        "${dst}"
  
    # Check if rsync was successful
    if [[ $? -ne 0 ]]; then
        echo "❌ Error: rsync failed for ${src} to ${dst}"
        exit 1
    fi

    echo ">>> Syncing data from ${src} to ${dst} DONE"
done

echo "Data sync completed successfully."
exit 0
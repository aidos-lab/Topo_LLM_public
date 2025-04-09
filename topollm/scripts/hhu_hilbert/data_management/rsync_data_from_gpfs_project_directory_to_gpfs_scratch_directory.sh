#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.

# Check if TOPO_LLM_REPOSITORY_BASE_PATH is set
if [[ -z "${TOPO_LLM_REPOSITORY_BASE_PATH}" ]]; then
    echo "Error: TOPO_LLM_REPOSITORY_BASE_PATH is not set."
    exit 1
fi

source "${TOPO_LLM_REPOSITORY_BASE_PATH}/.env"

if [[ -z "${GPFS_PROJECT_DIR_DSML}" ]]; then
    echo "Error: GPFS_PROJECT_DIR_DSML is not set."
    exit 1
fi

# Print variables
echo "TOPO_LLM_REPOSITORY_BASE_PATH=$TOPO_LLM_REPOSITORY_BASE_PATH"
echo "GPFS_PROJECT_DIR_DSML=$GPFS_PROJECT_DIR_DSML"

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Define directories

GPFS_PROJECT_PREFIX="/gpfs/project"
GPFS_SCRATCH_PREFIX="/gpfs/scratch"

# Note: Remember to add trailing slashes '/' to the directories.
RELATIVE_DIRS_TO_PROCESS_LIST=(
    # "ruppik/git-source/Topo_LLM/data/analysis/twonn.backup.2025-01-08/"
<<<<<<< HEAD
    # "ruppik/git-source/Topo_LLM/data/embeddings/arrays/"
    # "ruppik/git-source/Topo_LLM/data/embeddings/metadata/"
    "ruppik/git-source/Topo_LLM/data/models/"
=======
    "ruppik/git-source/Topo_LLM/data/embeddings/arrays/"
    "ruppik/git-source/Topo_LLM/data/embeddings/metadata/"
>>>>>>> dec81652c (PH)
)

for RELATIVE_DIR_TO_PROCESS in "${RELATIVE_DIRS_TO_PROCESS_LIST[@]}"; do
    SOURCE_DIR="${GPFS_PROJECT_PREFIX}/${RELATIVE_DIR_TO_PROCESS}"
    TARGET_DIR="${GPFS_SCRATCH_PREFIX}/${RELATIVE_DIR_TO_PROCESS}"

    echo ">>> Processing directory: ${RELATIVE_DIR_TO_PROCESS}"
    echo ">>> SOURCE_DIR=${SOURCE_DIR}"
    echo ">>> TARGET_DIR=${TARGET_DIR}"

    rsync \
        -avh --progress \
        "${SOURCE_DIR}" \
        "${TARGET_DIR}"
done


echo "Script finished."
exit $?

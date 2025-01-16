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


SOURCE_DIR="${GPFS_PROJECT_PREFIX}/ruppik/git-source/Topo_LLM/data/analysis/twonn.backup.2024-12-12.103000/"
TARGET_DIR="${GPFS_SCRATCH_PREFIX}/ruppik/git-source/Topo_LLM/data/analysis/twonn.backup.2024-12-12.103000/"


rsync \
  -avh --progress \
  "${SOURCE_DIR}" \
  "${TARGET_DIR}"


echo "Script finished."
exit $?

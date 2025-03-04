#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.

# This script copies data from the gpfs project directory to the gpfs user directory.

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
# Define target (actual data) and link (symlink) directories
DATA_DIR_SETSUMBT="${GPFS_PROJECT_DIR_DSML}/data/data-exp-eriments-zetsumbt"
USER_REPOSITORY_DIR_SETSUMBT="${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/setsumbt_checkpoints"

DATA_DIR_TRIPPY="${GPFS_PROJECT_DIR_DSML}/data/trippy-for-ben"
USER_REPOSITORY_DIR_TRIPPY="${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_checkpoints"

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Copy data.
#
# Example one-line copy command:
# `cp -r /gpfs/project/projects/dsml/data/setsumbt_experiments_new/multiwoz21 /gpfs/project/${USER}/git-source/Topo_LLM/data/models/setsumbt_checkpoints/multiwoz21`

# Check if the target directory already exists, if so, exit
if [[ -d "${USER_REPOSITORY_DIR_SETSUMBT}" ]]; then
  echo "Error: ${USER_REPOSITORY_DIR_SETSUMBT} already exists."
  exit 1
fi

echo "Copying data from ${DATA_DIR_SETSUMBT} to ${USER_REPOSITORY_DIR_SETSUMBT} ..."
cp -r \
  "${DATA_DIR_SETSUMBT}" \
  "${USER_REPOSITORY_DIR_SETSUMBT}"
echo "Copying data from ${DATA_DIR_SETSUMBT} to ${USER_REPOSITORY_DIR_SETSUMBT} DONE"


# Check if the target directory already exists, if so, exit
if [[ -d "${USER_REPOSITORY_DIR_TRIPPY}" ]]; then
  echo "Error: ${USER_REPOSITORY_DIR_TRIPPY} already exists."
  exit 1
fi

echo "Copying data from ${DATA_DIR_TRIPPY} to ${USER_REPOSITORY_DIR_TRIPPY} ..."
cp \
  "${DATA_DIR_TRIPPY}" \
  "${USER_REPOSITORY_DIR_TRIPPY}"
echo "Copying data from ${DATA_DIR_TRIPPY} to ${USER_REPOSITORY_DIR_TRIPPY} DONE"

echo "Copied data successfully."

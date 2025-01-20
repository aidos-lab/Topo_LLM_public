#!/bin/bash

# This script makes the following directories in the project's data directory:
# `data/models/setsumbt_checkpoints/`
# `data/models/trippy_checkpoints/`
# point to the corresponding folders containing the data in the 
# `/gpfs/project/projects/dsml/data`
# directory.

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

# Define target (actual data) and link (symlink) directories
DATA_DIR_SETSUMBT="${GPFS_PROJECT_DIR_DSML}/data/data-exp-eriments-zetsumbt"
LINK_DIR_SETSUMBT="${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/setsumbt_checkpoints"

DATA_DIR_TRIPPY="${GPFS_PROJECT_DIR_DSML}/data/trippy-for-ben"
LINK_DIR_TRIPPY="${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_checkpoints"

# Check if the directories or symlinks already exist
if [ -e "$LINK_DIR_SETSUMBT" ]; then
  echo "Warning: Directory or symlink '$LINK_DIR_SETSUMBT' already exists. Exiting."
  exit 1
fi

if [ -e "$LINK_DIR_TRIPPY" ]; then
  echo "Warning: Directory or symlink '$LINK_DIR_TRIPPY' already exists. Exiting."
  exit 1
fi

# Create symbolic links
echo "Linking setsumbt_checkpoints in git-source directory to data directory ..."
ln -s "$DATA_DIR_SETSUMBT" "$LINK_DIR_SETSUMBT"

echo "Linking trippy_checkpoints in git-source directory to data directory ..."
ln -s "$DATA_DIR_TRIPPY" "$LINK_DIR_TRIPPY"

echo "Symbolic links created successfully."

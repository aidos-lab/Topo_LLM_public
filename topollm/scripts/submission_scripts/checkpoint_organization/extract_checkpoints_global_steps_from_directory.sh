#!/bin/bash

# # # # # # # # # # # # # # # # # # # # # # # #
# Example usage:
#
# ./extract_checkpoints_global_steps_from_directory.sh "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/setsumbt_checkpoints/multiwoz21/roberta/setsumbt/gru/cosine/labelsmoothing/0.05/seed0/04-09-2024-14-53"

# Check if TOPO_LLM_REPOSITORY_BASE_PATH is set
if [[ -z "${TOPO_LLM_REPOSITORY_BASE_PATH}" ]]; then
  echo "Error: TOPO_LLM_REPOSITORY_BASE_PATH is not set."
  exit 1
fi

source "${TOPO_LLM_REPOSITORY_BASE_PATH}/.env"

# # # # # # # # # # # # # # # # # # # # # # # #
# Default directory (change this if needed)
DEFAULT_DIR="${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_checkpoints/results.42"

# Allow user to provide a directory as an argument, otherwise use the default
DIR=${1:-$DEFAULT_DIR}

# Ensure the directory exists
if [ ! -d "$DIR" ]; then
  echo "Error: Directory '$DIR' does not exist."
  exit 1
fi

# Explanation:
#
# - `ls "$DIR"`: Lists all files in the specified directory.
# - `awk -F'-' '/checkpoint-/ {print $2}'`: 
#     - Uses `-` as the field separator.
#     - Filters filenames that contain "checkpoint-".
#     - Extracts and prints the number after "checkpoint-".
# - `sort -n`: Sorts the extracted checkpoint numbers numerically.

ls "$DIR" | awk -F'-' '/checkpoint-/ {print $2}' | sort -n

# Exit with output of the last command
exit $?
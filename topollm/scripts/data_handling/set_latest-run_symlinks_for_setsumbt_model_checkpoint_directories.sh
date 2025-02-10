#!/bin/bash

# Check if TOPO_LLM_REPOSITORY_BASE_PATH is set
if [[ -z "${TOPO_LLM_REPOSITORY_BASE_PATH}" ]]; then
  echo "Error: TOPO_LLM_REPOSITORY_BASE_PATH is not set."
  exit 1
fi

source "${TOPO_LLM_REPOSITORY_BASE_PATH}/.env"

# Define the default base directory
DEFAULT_BASE_DIR="${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/setsumbt_checkpoints/multiwoz21/roberta/setsumbt/gru/cosine/labelsmoothing/0.05"

# Use the first argument as the base directory, or fallback to the default if not provided
BASE_DIR="${1:-$DEFAULT_BASE_DIR}"

# Check if the given base directory exists
if [ ! -d "$BASE_DIR" ]; then
  echo "@@@ Error: Directory '$BASE_DIR' does not exist."
  exit 1
fi

echo ">>> Setting latest symlinks in: $BASE_DIR"

# Loop through all seed directories (e.g., seed0, seed1, seed2)
for seed in "$BASE_DIR"/seed*; do
  echo ">>> Processing directory: $seed"
  
  # Ensure it is a directory
  if [ -d "$seed" ]; then
    # Find the most recent timestamped directory
    latest_subdir=$(ls -td "$seed"/*/ 2>/dev/null | head -n 1)

    # Check if we found a valid timestamped folder
    if [ -n "$latest_subdir" ]; then
      # Remove trailing slash from latest_subdir
      latest_subdir="${latest_subdir%/}"
      
      # Create or update the symbolic link
      ln -sfn "$latest_subdir" "$seed/latest-run"
      
      echo ">>> Symlink created: $seed/latest-run -> $latest_subdir"
    else
      echo "@@@ Warning: No timestamped subdirectories found in $seed"
    fi
  fi
done

echo ">>> Symlink setup complete."
exit 0

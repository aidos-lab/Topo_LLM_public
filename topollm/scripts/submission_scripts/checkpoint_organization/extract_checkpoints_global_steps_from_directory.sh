#!/bin/bash

# # # # # # # # # # # # # # # # # # # # # # # #
# Example usage:
#
# ./extract_checkpoints_global_steps_from_directory.sh --output-format "csv" --directory "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/setsumbt_checkpoints/multiwoz21/roberta/setsumbt/gru/cosine/labelsmoothing/0.05/seed0/04-09-2024-14-53"
#
# Expected output:
#
# > 2813,5626,8439,11252,14065,16878,19691,25317,33756,36569,39382,42195,50634,56260,70325,109707,115333,126585
#
# # # # # # # # # # # # # # # # # # # # # # # #
#
# Other examples:
#
# ./extract_checkpoints_global_steps_from_directory.sh --output-format "csv" --directory "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/setsumbt_checkpoints/multiwoz21/roberta/setsumbt/gru/cosine/labelsmoothing/0.05/seed1/latest-run"
#
# > 2813,5626,8439,11252,14065,16878,19691,25317,30943,42195,47821,53447,61886,109707
#
# ./extract_checkpoints_global_steps_from_directory.sh --output-format "csv" --directory "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/setsumbt_checkpoints/multiwoz21/roberta/setsumbt/gru/cosine/labelsmoothing/0.05/seed2/latest-run"
#
# > 2813,5626,8439,11252,14065,16878,19691,22504,28130,30943,33756,42195,50634,67512,106894,126585
# # # # # # # # # # # # # # # # # # # # # # # #

# Check if TOPO_LLM_REPOSITORY_BASE_PATH is set
if [[ -z "${TOPO_LLM_REPOSITORY_BASE_PATH}" ]]; then
  echo "Error: TOPO_LLM_REPOSITORY_BASE_PATH is not set."
  exit 1
fi

source "${TOPO_LLM_REPOSITORY_BASE_PATH}/.env"

# # # # # # # # # # # # # # # # # # # # # # # #
# Default directory (change this if needed)
DEFAULT_DIR="${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_checkpoints/results.42"
DIR="${DEFAULT_DIR}"
OUTPUT_FORMAT="list"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --directory)
      DIR="$2"; shift 2 ;;  # Set directory from next argument
    --output-format)
      if [[ "$2" == "csv" ]]; then
        OUTPUT_FORMAT="csv"
      elif [[ "$2" == "list" ]]; then
        OUTPUT_FORMAT="list"
      else
        echo "Error: Invalid output format. Use 'list' or 'csv'." >&2
        exit 1
      fi
      shift 2 ;;
    --help)
      echo "Usage: $0 [--directory <dir>] [--output-format csv|list]"
      exit 0 ;;
    *)
      echo "Error: Unknown option '$1'. Use --help for usage." >&2
      exit 1 ;;
  esac
done

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

# Extract checkpoint numbers and sort them
CHECKPOINTS=$(ls "$DIR" | awk -F'-' '/checkpoint-/ {print $2}' | sort -n)

# Ensure we have some checkpoints before processing
if [[ -z "$CHECKPOINTS" ]]; then
  echo "Error: No checkpoint files found in '$DIR'." >&2
  exit 1
fi

# Output in the requested format
if [[ "$OUTPUT_FORMAT" == "csv" ]]; then
  echo "$CHECKPOINTS" | paste -sd "," - 
else
  echo "$CHECKPOINTS"
fi

# Exit with output of the last command
exit $?
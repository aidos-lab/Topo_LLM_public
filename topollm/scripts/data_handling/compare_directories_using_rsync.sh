#!/bin/bash

# This script compares the contents of two directories using rsync's checksum option.
# It outputs a human-readable success message if the directories are identical,
# and a failure message with details if differences are found.
#
# Usage: ./compare_directories_using_rsync.sh

# # # # # # # # # # # # # # # # # # # # # # # # #
# Check if TOPO_LLM_REPOSITORY_BASE_PATH is set
if [[ -z "${TOPO_LLM_REPOSITORY_BASE_PATH}" ]]; then
  echo "❌ Error: TOPO_LLM_REPOSITORY_BASE_PATH is not set."
  exit 1
fi

# Load environment variables
source "${TOPO_LLM_REPOSITORY_BASE_PATH}/.env"

# Define source and destination directories.
# - The trailing slashes on the directory paths are important.
SOURCE="/Users/${USER}/Downloads/SentiX_Base_Model/"
DESTINATION="${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/EmoLoop/required_files/dst/SentiX_Base_Model/"

# # # # # # # # # # # # # # # # # # # # # # # # #
# Print variables
variables_to_log=(
  "SOURCE"
  "DESTINATION"
)

echo "📝 Printing bash script variables:"
for variable in "${variables_to_log[@]}"; do
    echo ">>> $variable=${!variable}"
done

# # # # # # # # # # # # # # # # # # # # # # # # #
# Syncing command

echo "🔍 Running rsync to compare directory contents..."
# Run rsync with options:
#   -r: Recursively compare directories.
#   -c: Compare files based on their checksum.
#   -n: Dry-run mode; no files are transferred.
#   -v: Verbose output; lists differences if any.
rsync -rcnv "$SOURCE" "$DESTINATION"

echo "🔍 Running rsync to compare directory contents DONE"

echo ""
echo "💡 Explanation: If no files are listed above, the folder contents agree perfectly!"

# Exit with the exit code of the rsync command
exit $?
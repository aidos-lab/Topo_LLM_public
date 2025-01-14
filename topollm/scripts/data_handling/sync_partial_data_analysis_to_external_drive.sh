#!/bin/bash

# Define source and destination directories
SOURCE="/Users/ruppik/git-source/Topo_LLM/data/analysis/twonn"
DESTINATION="/Volumes/ruppik_external/research_data/Topo_LLM/data/analysis/twonn"

# Parse arguments to optionally include --delete
DELETE_FLAG=""

if [[ "$1" == "--delete" ]]; then
    DELETE_FLAG="--delete"
    echo ">>> Delete flag is enabled. Files at destination not present in source will be deleted."
else
    echo ">>> Delete flag not enabled. Files at destination will not be deleted."
fi

# Run rsync with optional delete flag
rsync -avh \
    $DELETE_FLAG \
    "$SOURCE/" \
    "$DESTINATION/"

RSYNC_EXIT_CODE=$?
if [[ ${RSYNC_EXIT_CODE} -ne 0 ]]; then
    echo "@@@ Error: rsync failed with exit code ${RSYNC_EXIT_CODE}"
    exit ${RSYNC_EXIT_CODE}
fi

# Output success message
echo ">>> Synchronization from $SOURCE to $DESTINATION completed."
exit 0
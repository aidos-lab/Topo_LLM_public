#!/bin/bash

# Usage: ./copy_folder_exclude_large.sh source_folder target_folder

if [ "$#" -ne 2 ]; then
    echo ">>> Usage: $0 source_folder target_folder"
    exit 1
fi

SOURCE_FOLDER=$1
TARGET_FOLDER=$2

# Check if source folder exists
if [ ! -d "$SOURCE_FOLDER" ]; then
    echo "@@@ Error: Source folder '$SOURCE_FOLDER' does not exist."
    exit 1
fi

# Create the target folder if it doesn't exist
mkdir -p "$TARGET_FOLDER"

# Use rsync to copy files while excluding specific large files
rsync \
    -av \
    --exclude="local_estimates_pointwise_meta.pkl" \
    --exclude="array_for_estimator.npy" \
    "$SOURCE_FOLDER/" "$TARGET_FOLDER/"

RSYNC_EXIT_CODE=$?
if [[ ${RSYNC_EXIT_CODE} -ne 0 ]]; then
    echo "@@@ Error: rsync failed with exit code ${RSYNC_EXIT_CODE}"
    exit ${RSYNC_EXIT_CODE}
fi

echo ">>> Copy completed."
echo ">>> All files except the excluded ones have been copied to '$TARGET_FOLDER'."

exit 0
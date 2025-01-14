#!/bin/bash

# Print usage information
function usage() {
    echo "Usage: $0 [--dry-run] [--delete] pairs_file"
    echo "Options:"
    echo "  --dry-run   Perform a dry run without making changes."
    echo "  --delete    Delete files in the destination that are not present in the source."
    echo "Arguments:"
    echo "  pairs_file  Text file containing pairs of source and destination directories."
    echo "              Each line in the file should have the format: SOURCE DESTINATION"
    exit 1
}

# Trap SIGINT (Ctrl+C) to exit gracefully
trap "echo '>>> Script interrupted. Exiting.'; exit 1" SIGINT

# Ensure at least one argument is provided
if [[ $# -lt 1 ]]; then
    usage
fi

# Parse arguments
DRY_RUN_FLAG=""
DELETE_FLAG=""
PAIRS_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN_FLAG="--dry-run"
            echo ">>> Dry run mode is enabled. No changes will be made."
            ;;
        --delete)
            DELETE_FLAG="--delete"
            echo ">>> Delete flag is enabled. Files at destination not present in source will be deleted."
            ;;
        *)
            if [[ -z "$PAIRS_FILE" ]]; then
                PAIRS_FILE="$1"
            else
                echo "@@@ Error: Unexpected argument $1"
                usage
            fi
            ;;
    esac
    shift
done

# Check if pairs file is specified and exists
if [[ -z "$PAIRS_FILE" || ! -f "$PAIRS_FILE" ]]; then
    echo "@@@ Error: Invalid or missing pairs file."
    echo "PAIRS_FILE: $PAIRS_FILE"
    usage
fi

echo ">>> Using pairs file: $PAIRS_FILE"
echo ">>> Starting to read file ..."

# Read pairs from the file and process each one
while IFS=' ' read -r SOURCE DESTINATION || [[ -n "$SOURCE" ]]; do
    # Remove surrounding quotes if present
    SOURCE=$(eval echo "$SOURCE")
    DESTINATION=$(eval echo "$DESTINATION")

    if [[ -z "$SOURCE" || -z "$DESTINATION" ]]; then
        echo "@@@ Warning: Skipping invalid line in pairs file."
        continue
    fi

    echo ">>> Synchronizing from '$SOURCE' to '$DESTINATION' ..."

    rsync -avh \
        $DRY_RUN_FLAG \
        $DELETE_FLAG \
        "$SOURCE/" \
        "$DESTINATION/"

    RSYNC_EXIT_CODE=$?
    if [[ ${RSYNC_EXIT_CODE} -ne 0 ]]; then
        echo "@@@ Error: rsync failed with exit code ${RSYNC_EXIT_CODE}"
        exit ${RSYNC_EXIT_CODE}
    fi

    echo ">>> Synchronization from '$SOURCE' to '$DESTINATION' completed."
done < "$PAIRS_FILE"

echo ">>> Finished reading file."
echo ">>> All synchronizations completed successfully."

exit 0

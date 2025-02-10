#!/bin/bash

# # # # # # # # # # # # # # # # # # # # # # # # #
# This is a general script for syncing data from HHU Hilbert server to local machine.
#
# You can specify the list of subfolders to sync using the --folders option.
# You can also specify a file containing the list of subfolders to sync using the --file option.
#
# Example call:
# ./rsync_selected_directories_from_hhu_hilbert.sh --dry-run --folders "data/models/setsumbt_checkpoints/multiwoz21/roberta/setsumbt/gru/cosine/labelsmoothing/0.05/seed1/" 
# # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # #
# Function to print usage
usage() {
    echo "Usage: $0 [--dry-run] [--folders <folder1 folder2 ...>] [--file <file_with_folders>]"
    exit 1
}

# Default value for dry_run option
DRY_RUN_FLAG=""
SELECTED_SUBFOLDERS_LIST=()

# # # # # # # # # # # # # # # # # # # # # # # # #
# Parse command-line options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
        DRY_RUN_FLAG="--dry-run"
        shift
        ;;
        --folders)
        shift
        while [[ $# -gt 0 && $1 != --* ]]; do
            SELECTED_SUBFOLDERS_LIST+=("$1")
            shift
        done
        ;;
        --file)
        shift
        if [[ $# -gt 0 ]]; then
            if [[ -f $1 ]]; then
            while IFS= read -r line; do
                SELECTED_SUBFOLDERS_LIST+=("$line")
            done < "$1"
            else
            echo "@@@ Error: File $1 not found."
            exit 1
            fi
            shift
        else
            echo "@@@ Error: Missing file argument for --file."
            usage
        fi
        ;;
        *)
        echo "@@@ Error: Invalid option $1"
        usage
        ;;
    esac
done

# Ensure at least one folder is provided
if [[ ${#SELECTED_SUBFOLDERS_LIST[@]} -eq 0 ]]; then
    echo "@@@ Error: No folders specified."
    usage
fi

# Check if TOPO_LLM_REPOSITORY_BASE_PATH is set
if [[ -z "${TOPO_LLM_REPOSITORY_BASE_PATH}" ]]; then
    echo "@@@ Error: TOPO_LLM_REPOSITORY_BASE_PATH is not set."
    exit 1
fi

source "${TOPO_LLM_REPOSITORY_BASE_PATH}/.env"

# Print variables
echo ">>> TOPO_LLM_REPOSITORY_BASE_PATH=$TOPO_LLM_REPOSITORY_BASE_PATH"
echo ">>> ZIM_TOPO_LLM_REPOSITORY_BASE_PATH=$ZIM_TOPO_LLM_REPOSITORY_BASE_PATH"

# Print the list of selected subfolders
echo ">>> Selected subfolders:"
for SELECTED_SUBFOLDER in "${SELECTED_SUBFOLDERS_LIST[@]}"; do
    echo "<-----> $SELECTED_SUBFOLDER"
done

# ========================

SELECTED_SUBFOLDER_INDEX=0

for SELECTED_SUBFOLDER in "${SELECTED_SUBFOLDERS_LIST[@]}"; do
    echo "--------------------------------------------------------------------------------"
    echo ">>> Processing subfolder with index $SELECTED_SUBFOLDER_INDEX"
    echo ">>> Selected subfolder: $SELECTED_SUBFOLDER"

    SOURCE_DIR="Hilbert-Storage:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/$SELECTED_SUBFOLDER"
    TARGET_DIR="${TOPO_LLM_REPOSITORY_BASE_PATH}/$SELECTED_SUBFOLDER"

    # Create the target directory if it does not exist
    mkdir -p "$TARGET_DIR"

    # ========================

    echo ">>> Syncing data from HHU Hilbert server to local machine ..."
    echo ">>> SOURCE_DIR=$SOURCE_DIR"
    echo ">>> TARGET_DIR=$TARGET_DIR"

    rsync \
        -avhz \
        --progress \
        $DRY_RUN_FLAG \
        "${SOURCE_DIR}" \
        "${TARGET_DIR}"

    RSYNC_EXIT_CODE=$?
    if [[ ${RSYNC_EXIT_CODE} -ne 0 ]]; then
        echo "@@@ Error: rsync failed with exit code ${RSYNC_EXIT_CODE}"
        exit ${RSYNC_EXIT_CODE}
    fi

    # Increment the index
    SELECTED_SUBFOLDER_INDEX=$((SELECTED_SUBFOLDER_INDEX + 1))
    echo "--------------------------------------------------------------------------------"
done

echo ">>> rsync completed successfully."
exit 0

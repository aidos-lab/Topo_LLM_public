#!/bin/bash

# Initialize flag variables
DRY_RUN_OPTION="" # This option is passed to the rsync command
SYNC_TO_EXTERNAL_DRIVE=false

# Loop through arguments and process flags
for arg in "$@"; do
    case $arg in
        --sync_to_external_drive)
            SYNC_TO_EXTERNAL_DRIVE=true
            echo ">> Syncing to external hard drive."
            ;;
        --dry-run)
            DRY_RUN_OPTION="--dry-run" # This option is passed to the rsync command
            echo ">> Dry run."
            ;;
        *)
            echo "Unknown option: $arg" >&2 # redirect to stderr
            exit 1
            ;;
    esac
done

# Load the common variables
source "${TOPO_LLM_REPOSITORY_BASE_PATH}/topollm/scripts/hhu_hilbert/sync_data/common_variables.sh"

if [ "$SYNC_TO_EXTERNAL_DRIVE" = true ]; then
    TARGET_FOLDER_BASE_PATH="${EXTERNAL_DRIVE_TOPO_LLM_REPOSITORY_BASE_PATH}"
else
    TARGET_FOLDER_BASE_PATH="${TOPO_LLM_REPOSITORY_BASE_PATH}"
fi

# Print variables
echo "TOPO_LLM_REPOSITORY_BASE_PATH=$TOPO_LLM_REPOSITORY_BASE_PATH"
echo "ZIM_TOPO_LLM_REPOSITORY_BASE_PATH=$ZIM_TOPO_LLM_REPOSITORY_BASE_PATH"
echo "TARGET_FOLDER_BASE_PATH=$TARGET_FOLDER_BASE_PATH"


REPOSITORY_SUBDIRECTORY_PATH_LIST=(
    "data/analysis/prepared/"
    "data/analysis/twonn/"
)


for REPOSITORY_SUBDIRECTORY_PATH in "${REPOSITORY_SUBDIRECTORY_PATH_LIST[@]}"
do
    echo "==========================================================="
    echo "REPOSITORY_SUBDIRECTORY_PATH=$REPOSITORY_SUBDIRECTORY_PATH"
    
    # Following rsync instructions from:
    # https://wiki.hhu.de/pages/viewpage.action?pageId=55725648

    rsync -avz --progress $DRY_RUN_OPTION \
        "Hilbert-Storage:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/${REPOSITORY_SUBDIRECTORY_PATH}" \
        "${TARGET_FOLDER_BASE_PATH}/${REPOSITORY_SUBDIRECTORY_PATH}"

    echo "==========================================================="
done


# Exit with the exit code of the rsync command
exit $?
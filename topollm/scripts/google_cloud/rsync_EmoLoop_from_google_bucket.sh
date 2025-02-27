#!/bin/bash

DRY_RUN_FLAG=""

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --dry-run)
      # This is the dry-run flag which will show what file operations would be done without actually doing it
      DRY_RUN_FLAG="--dry-run"
      shift # Remove --dry-run from processing
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# # # # # # # # # # # # # # # # # # # # # # # #
# Check if TOPO_LLM_REPOSITORY_BASE_PATH is set
if [[ -z "${TOPO_LLM_REPOSITORY_BASE_PATH}" ]]; then
  echo "@@@ Error: TOPO_LLM_REPOSITORY_BASE_PATH is not set."
  exit 1
else
  echo ">>> TOPO_LLM_REPOSITORY_BASE_PATH is set to ${TOPO_LLM_REPOSITORY_BASE_PATH}"
fi

ENV_FILE_PATH="${TOPO_LLM_REPOSITORY_BASE_PATH}/.env"

echo ">>> ENV_FILE_PATH is set to ${ENV_FILE_PATH}"
echo ">>> Sourcing environment variables from ${ENV_FILE_PATH} ..."

source "${ENV_FILE_PATH}"

# # # # # # # # # # # # # # # # # #

SOURCE_PATH="gs://fengs/EmoLoop/required_files/dst"
DESTINATION_PATH="${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/EmoLoop/required_files/dst"


# # # # # # # # # # # # # # # # # #

echo ">>> Syncing from Google Cloud Storage bucket ..."


gcloud storage rsync -r \
    $DRY_RUN_FLAG \
    $SOURCE_PATH \
    $DESTINATION_PATH
    
if [[ $? -ne 0 ]]; then
    echo "@@@ Error: gcloud storage rsync failed."
    exit 1
fi

echo ">>> Data synchronization completed successfully."

echo ">>> Syncing from Google Cloud Storage bucket DONE"

# # # # # # # # #
# Exit script
echo ">>> Exiting script ..."
exit $?
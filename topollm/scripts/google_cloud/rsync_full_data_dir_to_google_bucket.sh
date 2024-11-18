#!/bin/bash

DRY_RUN_FLAG=""

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --dry_run)
      DRY_RUN_FLAG="-n"
      shift # Remove --dry_run from processing
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

echo ">>> Syncing data directory to Google Cloud Storage bucket ..."

# `-n` is the dry-run flag which will show what file operations would be done without actually doing it
gsutil -m rsync -r \
    $DRY_RUN_FLAG \
    $LOCAL_TOPO_LLM_DATA_DIR \
    $GC_TOPO_LLM_BUCKET_DATA_DIR
    

echo ">>> Syncing data directory to Google Cloud Storage bucket DONE"

# # # # # # # # #
# Exit script
echo ">>> Exiting script ..."
exit $?
#!/bin/bash

# Function to print usage
usage() {
  echo ">>> Usage: $0 [--dry-run]"
  exit 1
}

# Default value for dry_run option
DRY_RUN_FLAG=""

# Parse command-line options
if [[ $# -gt 1 ]]; then
  echo ">>> Error: Too many arguments."
  usage
fi

if [[ $# -eq 1 ]]; then
  case "$1" in
    --dry-run)
      DRY_RUN_FLAG="--dry-run"
      ;;
    *)
      echo ">>> Error: Invalid option $1"
      usage
      ;;
  esac
fi

# # # # # # # # # # # # # # # # # # # # # # # # #
# Check if TOPO_LLM_REPOSITORY_BASE_PATH is set
if [[ -z "${TOPO_LLM_REPOSITORY_BASE_PATH}" ]]; then
  echo ">>> Error: TOPO_LLM_REPOSITORY_BASE_PATH is not set."
  exit 1
fi

source "${TOPO_LLM_REPOSITORY_BASE_PATH}/.env"

# # # # # # # # # # # # # # # # # # # # # # # # #
# Print variables
echo ">>> TOPO_LLM_REPOSITORY_BASE_PATH=$TOPO_LLM_REPOSITORY_BASE_PATH"
echo ">>> ZIM_TOPO_LLM_REPOSITORY_BASE_PATH=$ZIM_TOPO_LLM_REPOSITORY_BASE_PATH"

REPOSITORY_SUBDIRECTORY_PATH_LIST=(
  "wandb_output_dir/"
)

# # # # # # # # # # # # # # # # # # # # # # # # #
# Following rsync instructions from:
# https://wiki.hhu.de/pages/viewpage.action?pageId=55725648

echo ">>> Syncing data from HHU Hilbert server to local machine ..."

for REPOSITORY_SUBDIRECTORY_PATH in "${REPOSITORY_SUBDIRECTORY_PATH_LIST[@]}"; do
  echo ">>> REPOSITORY_SUBDIRECTORY_PATH=$REPOSITORY_SUBDIRECTORY_PATH"

  rsync \
    -avz \
    --progress \
    $DRY_RUN_FLAG \
    "${ZIM_USERNAME}@Hilbert-Storage:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/${REPOSITORY_SUBDIRECTORY_PATH}" \
    "${TOPO_LLM_REPOSITORY_BASE_PATH}/${REPOSITORY_SUBDIRECTORY_PATH}"

  # Capture the exit code of rsync
  RSYNC_EXIT_CODE=$?
  if [[ ${RSYNC_EXIT_CODE} -ne 0 ]]; then
    echo ">>> Error: rsync failed with exit code ${RSYNC_EXIT_CODE}"
    exit ${RSYNC_EXIT_CODE}
  fi
done


echo ">>> All rsync commands completed successfully."
exit 0
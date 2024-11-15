#!/bin/bash

# # # # # # # # # # # # # # # # # # # # # # # # #
# Check if TOPO_LLM_REPOSITORY_BASE_PATH is set
if [[ -z "${TOPO_LLM_REPOSITORY_BASE_PATH}" ]]; then
  echo "Error: TOPO_LLM_REPOSITORY_BASE_PATH is not set."
  exit 1
fi

source "${TOPO_LLM_REPOSITORY_BASE_PATH}/.env"

# # # # # # # # # # # # # # # # # # # # # # # # #
# Print variables
echo "TOPO_LLM_REPOSITORY_BASE_PATH=$TOPO_LLM_REPOSITORY_BASE_PATH"
echo "ZIM_TOPO_LLM_REPOSITORY_BASE_PATH=$ZIM_TOPO_LLM_REPOSITORY_BASE_PATH"

REPOSITORY_SUBDIRECTORY_PATH="wandb_output_dir/"

# # # # # # # # # # # # # # # # # # # # # # # # #
# Following rsync instructions from:
# https://wiki.hhu.de/pages/viewpage.action?pageId=55725648
rsync -avz --progress \
    "${ZIM_USERNAME}@Hilbert-Storage:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/${REPOSITORY_SUBDIRECTORY_PATH}" \
    "${TOPO_LLM_REPOSITORY_BASE_PATH}/${REPOSITORY_SUBDIRECTORY_PATH}"

# Exit with the exit code of the rsync command
exit $?
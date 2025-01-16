#!/bin/bash

source "${TOPO_LLM_REPOSITORY_BASE_PATH}/.env"

# Print variables
echo "TOPO_LLM_REPOSITORY_BASE_PATH=$TOPO_LLM_REPOSITORY_BASE_PATH"
echo "ZIM_TOPO_LLM_REPOSITORY_BASE_PATH=$ZIM_TOPO_LLM_REPOSITORY_BASE_PATH"

REPOSITORY_SUBDIRECTORY_PATH="data/embeddings/perplexity/"

# Following rsync instructions from:
# https://wiki.hhu.de/pages/viewpage.action?pageId=55725648
rsync -avhz --progress \
    "Hilbert-Storage:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/${REPOSITORY_SUBDIRECTORY_PATH}" \
    "${TOPO_LLM_REPOSITORY_BASE_PATH}/${REPOSITORY_SUBDIRECTORY_PATH}"

# Exit with the exit code of the rsync command
exit $?
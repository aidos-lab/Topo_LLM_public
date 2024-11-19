#!/bin/bash

# Following rsync instructions from:
# https://wiki.hhu.de/pages/viewpage.action?pageId=55725648

echo "TOPO_LLM_REPOSITORY_BASE_PATH=$TOPO_LLM_REPOSITORY_BASE_PATH"

source "${TOPO_LLM_REPOSITORY_BASE_PATH}/.env"


rsync -avz --delete --progress \
    "${LOCAL_HUGGINGFACE_CACHE_PATH}" \
    "${ZIM_USERNAME}@Hilbert-Storage:/gpfs/project/${ZIM_USERNAME}/models/"

# Exit with the exit code of the rsync command
exit $?
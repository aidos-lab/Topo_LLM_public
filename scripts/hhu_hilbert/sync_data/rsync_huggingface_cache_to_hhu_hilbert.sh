#!/bin/bash

# Following rsync instructions from:
# https://wiki.hhu.de/pages/viewpage.action?pageId=55725648

source ./common_variables.sh

echo "TOPO_LLM_REPOSITORY_BASE_PATH=$TOPO_LLM_REPOSITORY_BASE_PATH"

LOCAL_HUGGINGFACE_CACHE_PATH="/Users/$LOCAL_USERNAME/.cache/huggingface"


rsync -avz --delete --progress \
    "${LOCAL_HUGGINGFACE_CACHE_PATH}" \
    "${ZIM_USERNAME}@storage.hpc.rz.uni-duesseldorf.de:/gpfs/project/${ZIM_USERNAME}/models/"

# Exit with the exit code of the rsync command
exit $?
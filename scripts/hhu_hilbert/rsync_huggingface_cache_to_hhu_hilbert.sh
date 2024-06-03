#!/bin/bash

# Following rsync instructions from:
# https://wiki.hhu.de/pages/viewpage.action?pageId=55725648

LOCAL_USERNAME="ruppik"
ZIM_USERNAME="ruppik"

echo "TOPO_LLM_REPOSITORY_BASE_PATH=$TOPO_LLM_REPOSITORY_BASE_PATH"

LOCAL_HUGGINGFACE_CACHE_PATH="/Users/$LOCAL_USERNAME/.cache/huggingface"


rsync -avz --delete --progress \
    "${LOCAL_HUGGINGFACE_CACHE_PATH}" \
    "${ZIM_USERNAME}@storage.hpc.rz.uni-duesseldorf.de:/gpfs/project/${ZIM_USERNAME}/models/"


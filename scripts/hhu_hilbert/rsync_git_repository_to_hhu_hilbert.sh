#!/bin/bash

# Following rsync instructions from:
# https://wiki.hhu.de/pages/viewpage.action?pageId=55725648

ZIM_USERNAME="ruppik"

echo "TOPO_LLM_REPOSITORY_BASE_PATH=$TOPO_LLM_REPOSITORY_BASE_PATH"

rsync -aPe ssh \
    "$TOPO_LLM_REPOSITORY_BASE_PATH/" \
    $ZIM_USERNAME@storage.hpc.rz.uni-duesseldorf.de:/gpfs/project/$ZIM_USERNAME/git-source/Topo_LLM/

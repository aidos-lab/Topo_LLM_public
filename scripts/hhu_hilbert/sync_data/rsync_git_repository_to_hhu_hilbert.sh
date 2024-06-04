#!/bin/bash

# Following rsync instructions from:
# https://wiki.hhu.de/pages/viewpage.action?pageId=55725648

source ./common_variables.sh

echo "TOPO_LLM_REPOSITORY_BASE_PATH=$TOPO_LLM_REPOSITORY_BASE_PATH"

EXCLUDE_FILE="rsync_git_repository_excludes.txt"

rsync -aPe ssh \
    --exclude-from="$EXCLUDE_FILE" \
    "${TOPO_LLM_REPOSITORY_BASE_PATH}/" \
    "${ZIM_USERNAME}@storage.hpc.rz.uni-duesseldorf.de:/gpfs/project/${ZIM_USERNAME}/git-source/Topo_LLM/"

# Exit with the exit code of the rsync command
exit $?
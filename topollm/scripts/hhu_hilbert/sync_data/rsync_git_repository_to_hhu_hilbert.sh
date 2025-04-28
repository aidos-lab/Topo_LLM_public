#!/bin/bash

source "${TOPO_LLM_REPOSITORY_BASE_PATH}/.env"

echo "TOPO_LLM_REPOSITORY_BASE_PATH=$TOPO_LLM_REPOSITORY_BASE_PATH"

# Following rsync instructions from:
# https://wiki.hhu.de/pages/viewpage.action?pageId=55725648
rsync -ahPe ssh \
    --exclude-from="$RSYNC_GIT_REPOSITORY_EXCLUDES_FILE" \
    "${TOPO_LLM_REPOSITORY_BASE_PATH}/" \
    "${ZIM_USERNAME}@${REMOTE_HOST}:/gpfs/project/${ZIM_USERNAME}/git-source/Topo_LLM/"

# Exit with the exit code of the rsync command
exit $?

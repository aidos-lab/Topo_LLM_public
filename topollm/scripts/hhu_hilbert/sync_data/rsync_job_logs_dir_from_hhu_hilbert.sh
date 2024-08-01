#!/bin/bash

source "${TOPO_LLM_REPOSITORY_BASE_PATH}/.env"

# Print variables
echo "TOPO_LLM_REPOSITORY_BASE_PATH=$TOPO_LLM_REPOSITORY_BASE_PATH"
echo "ZIM_TOPO_LLM_REPOSITORY_BASE_PATH=$ZIM_TOPO_LLM_REPOSITORY_BASE_PATH"


# Following rsync instructions from:
# https://wiki.hhu.de/pages/viewpage.action?pageId=55725648
rsync -avz --progress \
    "Hilbert-Storage:/gpfs/project/${ZIM_USERNAME}/job_logs/" \
    "${TOPO_LLM_REPOSITORY_BASE_PATH}/hilbert_job_logs/"

# Exit with the exit code of the rsync command
exit $?
#!/bin/bash

# Following rsync instructions from:
# https://wiki.hhu.de/pages/viewpage.action?pageId=55725648

echo "TOPO_LLM_REPOSITORY_BASE_PATH=$TOPO_LLM_REPOSITORY_BASE_PATH"

source "${TOPO_LLM_REPOSITORY_BASE_PATH}/topollm/scripts/hhu_hilbert/sync_data/common_variables.sh"

rsync -avz --progress \
    "${ZIM_USERNAME}@storage.hpc.rz.uni-duesseldorf.de:/gpfs/project/${ZIM_USERNAME}/job_logs/" \
    "${TOPO_LLM_REPOSITORY_BASE_PATH}/hilbert_job_logs/"

# Exit with the exit code of the rsync command
exit $?
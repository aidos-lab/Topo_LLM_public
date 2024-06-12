#!/bin/bash

# Following rsync instructions from:
# https://wiki.hhu.de/pages/viewpage.action?pageId=55725648

echo "TOPO_LLM_REPOSITORY_BASE_PATH=$TOPO_LLM_REPOSITORY_BASE_PATH"

source "${TOPO_LLM_REPOSITORY_BASE_PATH}/scripts/hhu_hilbert/sync_data/common_variables.sh"

REPOSITORY_SUBDIRECTORY_PATH="wandb_output_dir/"

rsync -avz --progress \
    "${ZIM_USERNAME}@storage.hpc.rz.uni-duesseldorf.de:/gpfs/project/${ZIM_USERNAME}/git-source/Topo_LLM/${REPOSITORY_SUBDIRECTORY_PATH}" \
    "${TOPO_LLM_REPOSITORY_BASE_PATH}/${REPOSITORY_SUBDIRECTORY_PATH}"

# Exit with the exit code of the rsync command
exit $?
#!/bin/bash

# Following rsync instructions from:
# https://wiki.hhu.de/pages/viewpage.action?pageId=55725648

echo "TOPO_LLM_REPOSITORY_BASE_PATH=$TOPO_LLM_REPOSITORY_BASE_PATH"

source "${TOPO_LLM_REPOSITORY_BASE_PATH}/.env"

LOCAL_NLTK_DATA_PATH="$HOME/nltk_data/"

rsync -avhz --delete --progress \
    "${LOCAL_NLTK_DATA_PATH}" \
    "${ZIM_USERNAME}@Hilbert-Storage:/home/${ZIM_USERNAME}/nltk_data/"

# Exit with the exit code of the rsync command
exit $?
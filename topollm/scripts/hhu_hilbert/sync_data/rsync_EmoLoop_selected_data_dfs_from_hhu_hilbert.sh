#!/bin/bash

# # # # # # # # # # # # # # # # # # # # # # # # #
# Check if variables are set
if [[ -z "${TOPO_LLM_REPOSITORY_BASE_PATH}" ]]; then
  echo "❌ Error: TOPO_LLM_REPOSITORY_BASE_PATH is not set."
  exit 1
fi
if [[ -z "${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}" ]]; then
  echo "❌ Error: ZIM_TOPO_LLM_REPOSITORY_BASE_PATH is not set."
  exit 1
fi

# The rsync command uses the following options:
#   - '-a': Archive mode to preserve file permissions, timestamps, and symbolic links.
#   - '-v': Verbose mode to show detailed output.
#   - '-z': Compress file data during the transfer.
#
# The source and destination paths are set to the same relative path on the remote and local machines.
rsync -avz \
    --include='*/' \
    --include='df_*.csv' \
    --exclude='*' \
    Hilbert-Storage:"${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/EmoLoop/output_dir/" \
    "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/EmoLoop/output_dir/"
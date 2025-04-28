#!/bin/bash

source "${TOPO_LLM_REPOSITORY_BASE_PATH}/.env"

# # # # # # # # # # # # # # # # # # # # # # # # #
# Check if variables are set
check_variable() {
  local var_name="$1"
  local var_value="${!var_name}"
  if [[ -z "${var_value}" ]]; then
    echo "❌ Error: ${var_name} is not set."
    exit 1
  else
    echo "✅ ${var_name}=${var_value}"
  fi
}

check_variable "TOPO_LLM_REPOSITORY_BASE_PATH"
check_variable "ZIM_TOPO_LLM_REPOSITORY_BASE_PATH"
check_variable "REMOTE_HOST"

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
  "${REMOTE_HOST}:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/EmoLoop/output_dir/" \
  "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/EmoLoop/output_dir/"

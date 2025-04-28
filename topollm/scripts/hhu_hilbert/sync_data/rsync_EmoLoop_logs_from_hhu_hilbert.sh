#!/bin/bash

# # # #
# This command synchronizes only the log.txt files from the remote machine.

source "${TOPO_LLM_REPOSITORY_BASE_PATH}/.env"

# The rsync command uses the following options:
#   - '-a': Archive mode to preserve file permissions, timestamps, and symbolic links.
#   - '-v': Verbose mode to show detailed output.
#   - '-z': Compress file data during the transfer.
#
# The include/exclude patterns are defined as follows:
#   --include='*/'
#       Includes all directories so that rsync can traverse the entire folder structure.
#   --include='log.txt'
#       Includes any file named 'log.txt' found in the directories.
#   --exclude='*'
#       Excludes all other files not explicitly included above.
#
# The source and destination paths are set to the same relative path on the remote and local machines.
rsync -avz \
    --include='*/' \
    --include='log.txt' \
    --exclude='*' \
    "${REMOTE_HOST}:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/EmoLoop/output_dir/" \
    "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/EmoLoop/output_dir/"

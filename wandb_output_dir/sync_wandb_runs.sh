#!/bin/bash

WANDB_OUTPUT_DIR_PATH="${TOPO_LLM_REPOSITORY_BASE_PATH}/wandb_output_dir"

wandb sync --include-offline "${WANDB_OUTPUT_DIR_PATH}/wandb/offline-*"

# Note: We usually do not need to sync the online runs, because they are already synced automatically.
#
# wandb sync --include-offline "${WANDB_OUTPUT_DIR_PATH}/wandb/run-*"

# Exit with the exit status of the last command
exit $?
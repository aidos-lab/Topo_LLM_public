#!/bin/bash

echo "TOPO_LLM_REPOSITORY_BASE_PATH=$TOPO_LLM_REPOSITORY_BASE_PATH"

# Check if TOPO_LLM_REPOSITORY_BASE_PATH is set
if [[ -z "${TOPO_LLM_REPOSITORY_BASE_PATH}" ]]; then
    echo "Error: TOPO_LLM_REPOSITORY_BASE_PATH is not set."
    exit 1
fi

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
check_variable "REMOTE_HOST"
check_variable "ZIM_USERNAME"
check_variable "ZIM_TOPO_LLM_REPOSITORY_BASE_PATH"

# Check for the dry run option
DRY_RUN=false

if [[ "$1" == "--dry_run" ]]; then
    DRY_RUN=true
    echo "Dry run mode enabled. Listing directories without syncing."
fi

# Use SSH to run the find command on the remote server and list all "wandb" directories
wandb_dirs=$(
    ssh \
        "${ZIM_USERNAME}@${REMOTE_HOST}" \
        "find ${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/ -type d -name 'wandb'"
)

# Iterate through each found "wandb" directory and sync it
for remote_dir in $wandb_dirs; do
    # Construct the local directory path
    local_dir="${TOPO_LLM_REPOSITORY_BASE_PATH}/${remote_dir#/gpfs/project/${ZIM_USERNAME}/git-source/Topo_LLM/}"

    if [[ "$DRY_RUN" == true ]]; then
        # If dry run, just print the paths
        echo ">>> Dry run: Would sync ${ZIM_USERNAME}@${REMOTE_HOST}:${remote_dir} to ${local_dir} (if not in exclude list)."
    else
        echo ">>> Syncing ${ZIM_USERNAME}@${REMOTE_HOST}:${remote_dir} to ${local_dir}"
        # Actual sync command
        rsync -avhz --progress \
            --exclude-from="$RSYNC_GIT_REPOSITORY_EXCLUDES_FILE" \
            "${ZIM_USERNAME}@${REMOTE_HOST}:${remote_dir}/" \
            "${local_dir}/"
    fi
done

# Exit with the exit code of the last rsync command
exit $?

#!/bin/bash

echo "TOPO_LLM_REPOSITORY_BASE_PATH=$TOPO_LLM_REPOSITORY_BASE_PATH"

# Check if TOPO_LLM_REPOSITORY_BASE_PATH is set
if [[ -z "${TOPO_LLM_REPOSITORY_BASE_PATH}" ]]; then
  echo "Error: TOPO_LLM_REPOSITORY_BASE_PATH is not set."
  exit 1
fi

source "${TOPO_LLM_REPOSITORY_BASE_PATH}/.env"

# Check for the dry run option
DRY_RUN=false

if [[ "$1" == "--dry_run" ]]; then
    DRY_RUN=true
    echo "Dry run mode enabled. Listing directories without syncing."
fi

# Use SSH to run the find command on the remote server and list all "wandb" directories
wandb_dirs=$(
    ssh \
    "${ZIM_USERNAME}@Hilbert-Storage" \
    "find ${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/ -type d -name 'wandb'"
)

# Iterate through each found "wandb" directory and sync it
for remote_dir in $wandb_dirs; do
    # Construct the local directory path
    local_dir="${TOPO_LLM_REPOSITORY_BASE_PATH}/${remote_dir#/gpfs/project/${ZIM_USERNAME}/git-source/Topo_LLM/}"
    
    if [[ "$DRY_RUN" == true ]]; then
        # If dry run, just print the paths
        echo ">>> Dry run: Would sync ${ZIM_USERNAME}@Hilbert-Storage:${remote_dir} to ${local_dir} (if not in exclude list)."
    else
        echo ">>> Syncing ${ZIM_USERNAME}@Hilbert-Storage:${remote_dir} to ${local_dir}"
        # Actual sync command
        rsync -avz --progress \
            --exclude-from="$RSYNC_GIT_REPOSITORY_EXCLUDES_FILE" \
            "${ZIM_USERNAME}@Hilbert-Storage:${remote_dir}/" \
            "${local_dir}/"
    fi
done

# Exit with the exit code of the last rsync command
exit $?

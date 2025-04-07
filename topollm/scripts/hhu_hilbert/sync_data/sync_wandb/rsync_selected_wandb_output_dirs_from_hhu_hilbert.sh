#!/bin/bash

# Function to print usage
usage() {
    echo ">>> Usage: $0 [--dry-run]"
    exit 1
}

# Default value for dry_run option
DRY_RUN_FLAG=""

# Parse command-line options
if [[ $# -gt 1 ]]; then
    echo ">>> Error: Too many arguments."
    usage
fi

if [[ $# -eq 1 ]]; then
    case "$1" in
    --dry-run)
        DRY_RUN_FLAG="--dry-run"
        ;;
    *)
        echo ">>> Error: Invalid option $1"
        usage
        ;;
    esac
fi

# # # # # # # # # # # # # # # # # # # # # # # # #
# Check if TOPO_LLM_REPOSITORY_BASE_PATH is set
if [[ -z "${TOPO_LLM_REPOSITORY_BASE_PATH}" ]]; then
    echo ">>> Error: TOPO_LLM_REPOSITORY_BASE_PATH is not set."
    exit 1
fi

source "${TOPO_LLM_REPOSITORY_BASE_PATH}/.env"

# # # # # # # # # # # # # # # # # # # # # # # # #
# Print variables
echo ">>> TOPO_LLM_REPOSITORY_BASE_PATH=$TOPO_LLM_REPOSITORY_BASE_PATH"

REPOSITORY_SUBDIRECTORY_PATH_LIST=(
    "wandb_output_dir/Topo_LLM_finetuning_from_submission_script_for_5_epochs_and_linear_lr_schedule/"
    "wandb_output_dir/Topo_LLM_gpt2_finetuning_from_submission_script_for_5_epochs_and_linear_lr_schedule/"
    "wandb_output_dir/Topo_LLM_roberta-base_finetuning_from_submission_script_for_5_epochs_and_linear_lr_schedule_freeze_lm_head/"
    "wandb_output_dir/Topo_LLM_roberta-base_finetuning_for_ep-5_lr-linear_no_freeze_different_data_subsampling_seeds/"
)

SCRIPT_PATH="${TOPO_LLM_REPOSITORY_BASE_PATH}/topollm/scripts/hhu_hilbert/sync_data/rsync_selected_directories_from_hhu_hilbert.sh"

${SCRIPT_PATH} \
    ${DRY_RUN_FLAG} \
    --folders "${REPOSITORY_SUBDIRECTORY_PATH_LIST[@]}"

exit 0

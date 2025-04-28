#!/bin/bash

# # # # # # # # # # # # # # # # # # # # # # # # #
# NOTE:
# This script syncs selected local estimates metadata from the HHU Hilbert server to the local machine.

# # # # # # # # # # # # # # # # # # # # # # # # #
# Default values
DRY_RUN_FLAG=""

# Example path to one of the local estimates metadata files:
# 'data/analysis/local_estimates/data=multiwoz21_rm-empty=True_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags/split=validation_samples=10000_sampling=random_sampling-seed=778/edh-mode=regular_lvl=token/add-prefix-space=False_max-len=512/model=roberta-base-trippy_multiwoz21_seed-42_ckpt-3549_task=masked_lm_dr=defaults/layer=-1_agg=mean/norm=None/sampling=random_seed=42_samples=150000/desc=twonn_samples=60000_zerovec=keep_dedup=array_deduplicator_noise=do_nothing/n-neighbors-mode=absolute_size_n-neighbors=128/local_estimates_pointwise_array.npy

REMOTE_BASE_DIR="${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/data/analysis/local_estimates/"

# Note that we are only matching the -1 layer metadata files,
# and that other parameters are fixed as well in this pattern.
PATH_TO_MATCH="*data=*/split=*/edh-mode=regular_lvl=token/add-prefix-space=False_max-len=512/model=roberta-base-trippy_multiwoz21_seed-42_ckpt-*_task=masked_lm_dr=defaults/layer=-1_agg=mean/norm=None/sampling=random_seed=42_samples=150000/desc=twonn_samples=60000_zerovec=keep_dedup=array_deduplicator_noise=do_nothing/n-neighbors-mode=absolute_size_n-neighbors=128/local_estimates_pointwise_meta.pkl"

TARGET_DIR="${TOPO_LLM_REPOSITORY_BASE_PATH}/data/analysis/local_estimates/"

# # # # # # # # # # # # # # # # # # # # # # # # #
# Function to print usage
usage() {
    echo "💡 Usage: $0 [--dry-run]"
    exit 1
}

# Parse command-line options
if [[ $# -gt 1 ]]; then
    echo "❌ Error: Too many arguments."
    usage
fi

if [[ $# -eq 1 ]]; then
    case "$1" in
    --dry-run)
        DRY_RUN_FLAG="--dry-run"
        ;;
    *)
        echo "❌ Error: Invalid option $1"
        usage
        ;;
    esac
fi

# # # # # # # # # # # # # # # # # # # # # # # # #
# Check if TOPO_LLM_REPOSITORY_BASE_PATH is set
if [[ -z "${TOPO_LLM_REPOSITORY_BASE_PATH}" ]]; then
    echo "❌ Error: TOPO_LLM_REPOSITORY_BASE_PATH is not set."
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
check_variable "ZIM_TOPO_LLM_REPOSITORY_BASE_PATH"

# # # # # # # # # # # # # # # # # # # # # # # # #
# Print variables
variables_to_log=(
    "TOPO_LLM_REPOSITORY_BASE_PATH"
    "ZIM_TOPO_LLM_REPOSITORY_BASE_PATH"
    "REMOTE_HOST"
    "REMOTE_BASE_DIR"
    "PATH_TO_MATCH"
    "TARGET_DIR"
)

echo "📝 Printing bash script variables:"
for variable in "${variables_to_log[@]}"; do
    echo ">>> $variable=${!variable}"
done

# ======================================================================================== #
# Find the matching files on the remote machine
# ======================================================================================== #

# Find matching files on the remote machine.
#
# Explanation of the find command:
# - `find $REMOTE_BASE_DIR -type f -path '$PATH_TO_MATCH'` searches for matching files.
#   - `-type f`: Ensures only files (not directories) are selected.
#   - `-path '$PATH_TO_MATCH'`: Filters files based on the provided pattern.
#
# Explanation of the sed command:
# - `sed 's|^$REMOTE_BASE_DIR||'` removes the `$REMOTE_BASE_DIR` prefix from each path.
# - `s|...|...|` is a substitution command in `sed`:
#   - `s` means "substitute".
#   - `|` is used as a delimiter instead of `/` to avoid conflicts with file paths.
# - `^$REMOTE_BASE_DIR` ensures that we only remove `$REMOTE_BASE_DIR` **if it appears at the start** (this is indicated by the caret ^).
# - The replacement is empty, meaning that `$REMOTE_BASE_DIR` is removed, leaving a **relative path**.
# - This is necessary for `rsync --relative` to reconstruct the correct directory structure inside `$TARGET_DIR`
#   instead of placing everything under `/`.
echo "🔍 Searching for matching files on remote machine ($REMOTE_HOST) ..."
MATCHED_FILES=$(ssh "$REMOTE_HOST" "find $REMOTE_BASE_DIR -type f -path '$PATH_TO_MATCH' | sed 's|^$REMOTE_BASE_DIR||'")

# Check if any files were found
if [[ -z "$MATCHED_FILES" ]]; then
    echo "❌ No matching files found!"
    exit 1
fi

# Print matched files for debugging
MATCHED_FILE_COUNT=$(echo "$MATCHED_FILES" | wc -l)
echo "✅ Total matched files: $MATCHED_FILE_COUNT"
echo "✅ Printing list of matched files:"
echo "$MATCHED_FILES"
echo "✅ Done printing list of $MATCHED_FILE_COUNT matched files."

# ======================================================================================== #
# Run rsync to sync the matched files to the local machine
# ======================================================================================== #

# Create the target directory if it does not exist
mkdir -p "$TARGET_DIR"

# Perform the actual rsync
#
# Following rsync instructions from:
# https://wiki.hhu.de/pages/viewpage.action?pageId=55725648
#
# Explanation:
# - `--files-from=-`: Takes the list of files from standard input (provided by `echo "$MATCHED_FILES"`).
# - `--relative`: Ensures rsync reconstructs the directory hierarchy **relative** to `$REMOTE_BASE_DIR`.
# - `$REMOTE_HOST:$REMOTE_BASE_DIR`: Tells `rsync` where the files are located.
# - `$TARGET_DIR/`: Defines where the files should be placed locally.

echo "🚀 Starting file sync ..."

echo "$MATCHED_FILES" | rsync \
    -avhz --progress $DRY_RUN_FLAG \
    --files-from=- \
    --relative \
    "$REMOTE_HOST:$REMOTE_BASE_DIR" \
    "$TARGET_DIR/"

echo "🔁 rsync command completed."

# Capture the exit code of rsync
RSYNC_EXIT_CODE=$?
if [[ ${RSYNC_EXIT_CODE} -ne 0 ]]; then
    echo "❌ Error: rsync failed with exit code ${RSYNC_EXIT_CODE}"
    exit ${RSYNC_EXIT_CODE}
fi

# ======================================================================================== #

echo "✅ rsync completed successfully."
exit 0

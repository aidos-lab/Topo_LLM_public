#!/bin/bash

# Function to print usage
usage() {
  echo "@@@ Usage: $0 [--dry-run]"
  exit 1
}

# Default value for dry_run option
DRY_RUN_FLAG=""

# Parse command-line options
if [[ $# -gt 1 ]]; then
  echo "@@@ Error: Too many arguments."
  usage
fi

if [[ $# -eq 1 ]]; then
  case "$1" in
    --dry-run)
      DRY_RUN_FLAG="--dry-run"
      ;;
    *)
      echo "Error: Invalid option $1"
      usage
      ;;
  esac
fi


# Check if TOPO_LLM_REPOSITORY_BASE_PATH is set
if [[ -z "${TOPO_LLM_REPOSITORY_BASE_PATH}" ]]; then
  echo "@@@ Error: TOPO_LLM_REPOSITORY_BASE_PATH is not set."
  exit 1
fi

source "${TOPO_LLM_REPOSITORY_BASE_PATH}/.env"

# Print variables
echo ">>> TOPO_LLM_REPOSITORY_BASE_PATH=$TOPO_LLM_REPOSITORY_BASE_PATH"
echo ">>> ZIM_TOPO_LLM_REPOSITORY_BASE_PATH=$ZIM_TOPO_LLM_REPOSITORY_BASE_PATH"

# # # #
# NOTE: 
# This script is only syncing a selected subdirectory to the local machine.


SELECTED_SUBFOLDERS_LIST=(
  # "data/analysis/prepared/data-multiwoz21_split-validation_ctxt-dataset_entry_samples-3000_feat-col-ner_tags/lvl-token/add-prefix-space-True_max-len-512/model-roberta-base_task-masked_lm/"
  "data/analysis/twonn/data=wikitext-103-v1_rm-empty=True_spl-mode=proportions_spl-shuf=True_spl-seed=0_tr=0.8_va=0.1_te=0.1_ctxt=dataset_entry_feat-col=ner_tags/split=train_samples=10000_sampling=random_sampling-seed=778"
)

for SELECTED_SUBFOLDER in "${SELECTED_SUBFOLDERS_LIST[@]}"; do
  echo ">>> Selected subfolder: $SELECTED_SUBFOLDER"

  SOURCE_DIR="Hilbert-Storage:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}/$SELECTED_SUBFOLDER"
  TARGET_DIR="${TOPO_LLM_REPOSITORY_BASE_PATH}/$SELECTED_SUBFOLDER"

  # Create the target directory if it does not exist
  mkdir -p "$TARGET_DIR"

  # ========================

  echo ">>> Syncing data from HHU Hilbert server to local machine ..."
  echo ">>> SOURCE_DIR=$SOURCE_DIR"
  echo ">>> TARGET_DIR=$TARGET_DIR"

  # Following rsync instructions from:
  # https://wiki.hhu.de/pages/viewpage.action?pageId=55725648
  rsync \
    -avz \
    --progress \
    $DRY_RUN_FLAG \
    "${SOURCE_DIR}" \
    "${TARGET_DIR}"

  # Capture the exit code of rsync
  RSYNC_EXIT_CODE=$?
  if [[ ${RSYNC_EXIT_CODE} -ne 0 ]]; then
    echo "@@@ Error: rsync failed with exit code ${RSYNC_EXIT_CODE}"
    exit ${RSYNC_EXIT_CODE}
  fi
done

echo ">>> rsync completed successfully."
exit 0
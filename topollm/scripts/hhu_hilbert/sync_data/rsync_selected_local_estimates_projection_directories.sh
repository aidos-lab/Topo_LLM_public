#!/bin/bash

# Check if TOPO_LLM_REPOSITORY_BASE_PATH is set
if [[ -z "${TOPO_LLM_REPOSITORY_BASE_PATH}" ]]; then
    echo "@@@ Error: TOPO_LLM_REPOSITORY_BASE_PATH is not set."
    exit 1
fi

source "${TOPO_LLM_REPOSITORY_BASE_PATH}/.env"

SCRIPT_PATH="${TOPO_LLM_REPOSITORY_BASE_PATH}/topollm/scripts/hhu_hilbert/sync_data/rsync_selected_directories_from_cluster_backend.sh"

# Optional flags for the inner script
DRY_RUN_FLAG=""

# Parse simple CLI flags for convenience
while [[ $# -gt 0 ]]; do
    case "$1" in
    --dry-run)
        DRY_RUN_FLAG="--dry-run"
        shift
        ;;
    *)
        echo "❌ Unknown argument: $1"
        echo "Usage: $0 [--dry-run]"
        exit 1
        ;;
    esac
done

DATA_DIRECTORY_LIST=(
    # "data=iclr_2024_submissions_rm-empty=True_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags"
    # "data=multiwoz21_rm-empty=True_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags"
    # "data=one-year-of-tsla-on-reddit_rm-empty=True_spl-mode=proportions_spl-shuf=True_spl-seed=0_tr=0.8_va=0.1_te=0.1_ctxt=dataset_entry_feat-col=ner_tags"
    # "data=sgd_rm-empty=True_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags"
    # "data=wikitext-103-v1_strip-True_rm-empty=True_spl-mode=proportions_spl-shuf=True_spl-seed=0_tr=0.8_va=0.1_te=0.1_ctxt=dataset_entry_feat-col=ner_tags"
    "data=luster_column=source_rm-empty=True_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags"
    "data=luster_column=source_target_rm-empty=True_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags"
)

SPLIT_DIRECTORY_LIST=(
    "split=validation_samples=7000_sampling=random_sampling-seed=778"
    "split=validation_samples=10000_sampling=random_sampling-seed=778"
)

MODEL_DIRECTORY_LIST=(
    "model=Phi-3.5-mini-instruct_task=causal_lm_dr=defaults"
)


for DATA_DIRECTORY in "${DATA_DIRECTORY_LIST[@]}"; do
    for SPLIT_DIRECTORY in "${SPLIT_DIRECTORY_LIST[@]}"; do
        for MODEL_DIRECTORY in "${MODEL_DIRECTORY_LIST[@]}"; do
            SELECTED_SUBFOLDER="data/"
            SELECTED_SUBFOLDER+="saved_plots/local_estimates_projection/"
            SELECTED_SUBFOLDER+="$DATA_DIRECTORY/"
            SELECTED_SUBFOLDER+="$SPLIT_DIRECTORY/"
            SELECTED_SUBFOLDER+="edh-mode=regular_lvl=token/add-prefix-space=False_max-len=512/"
            SELECTED_SUBFOLDER+="$MODEL_DIRECTORY/"

            echo "=========================="
            echo "Syncing selected directory: ${SELECTED_SUBFOLDER}"
            echo "=========================="

            ${SCRIPT_PATH} \
                ${DRY_RUN_FLAG} \
                --folders \
                "${SELECTED_SUBFOLDER}"
        done
    done
done

# End of script
echo "Sync completed for selected directories."
echo "Exiting script with return code $?"

exit $?

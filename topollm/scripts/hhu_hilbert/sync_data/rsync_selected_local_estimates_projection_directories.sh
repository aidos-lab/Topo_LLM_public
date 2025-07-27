#!/bin/bash

# Check if TOPO_LLM_REPOSITORY_BASE_PATH is set
if [[ -z "${TOPO_LLM_REPOSITORY_BASE_PATH}" ]]; then
    echo "@@@ Error: TOPO_LLM_REPOSITORY_BASE_PATH is not set."
    exit 1
fi

source "${TOPO_LLM_REPOSITORY_BASE_PATH}/.env"

SCRIPT_PATH="${TOPO_LLM_REPOSITORY_BASE_PATH}/topollm/scripts/hhu_hilbert/sync_data/rsync_selected_directories_from_cluster_backend.sh"

DATA_DIRECTORY_LIST=(
    "data=iclr_2024_submissions_rm-empty=True_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags"
    "data=multiwoz21_rm-empty=True_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags"
    "data=one-year-of-tsla-on-reddit_rm-empty=True_spl-mode=proportions_spl-shuf=True_spl-seed=0_tr=0.8_va=0.1_te=0.1_ctxt=dataset_entry_feat-col=ner_tags"
    "data=sgd_rm-empty=True_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags"
    "data=wikitext-103-v1_strip-True_rm-empty=True_spl-mode=proportions_spl-shuf=True_spl-seed=0_tr=0.8_va=0.1_te=0.1_ctxt=dataset_entry_feat-col=ner_tags"
)

for DATA_DIRECTORY in "${DATA_DIRECTORY_LIST[@]}"; do
    SELECTED_SUBFOLDER="data/saved_plots/local_estimates_projection/$DATA_DIRECTORY/split=validation_samples=10000_sampling=random_sampling-seed=778/edh-mode=regular_lvl=token/add-prefix-space=False_max-len=512/model=Phi-3.5-mini-instruct_task=masked_lm_dr=defaults/"
    
    echo "=========================="
    echo "Syncing selected directory: ${SELECTED_SUBFOLDER}"
    echo "=========================="

    ${SCRIPT_PATH} \
        --folders \
        "${SELECTED_SUBFOLDER}"
done

# End of script
echo "Sync completed for selected directories."
echo "Exiting script with return code $?"

exit $?
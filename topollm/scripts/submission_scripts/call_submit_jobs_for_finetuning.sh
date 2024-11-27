#!/bin/bash

# ================================================================== #
# Start submission script
echo ">>> Submission script started."
# ================================================================== #

# LOCAL_OPTION="--local"

# Leave DRY_RUN_OPTION empty to run without the dry-run option, i.e., to actually submit the jobs
#
# DRY_RUN_OPTION="--dry-run"

# RUN_ONLY_SELECTED_CONFIGS_OPTION="run_all"
# RUN_ONLY_SELECTED_CONFIGS_OPTION="run_single_random"
RUN_ONLY_SELECTED_CONFIGS_OPTION="run_only_first"

DROPOUT_PARAMETER_LIST=(
  "0.05"
  "0.2"
  "0.25"
  "0.3"
  "0.35"
)

### Finetuning
#

for DROPOUT_PARAMETER in "${DROPOUT_PARAMETER_LIST[@]}"; do
  poetry run submit_jobs \
    --experiment-selector multiwoz21_different_data_subsampling_number_of_samples \
    --experiment-stage skip_compute_embeddings_and_multiple_pipeline_runs \
    --finetuning-datasets-list-option "multiwoz21_and_reddit_small" \
    --wandb-project "Topo_LLM_finetuning_from_submission_script_dropout_different_choices" \
    --additional-overrides "+finetuning.base_model.dropout.mode=modify_roberta_dropout_parameters" \
    --additional-overrides "+finetuning.base_model.dropout.probabilities.hidden_dropout_prob=${DROPOUT_PARAMETER}" \
    --additional-overrides "+finetuning.base_model.dropout.probabilities.attention_probs_dropout_prob=${DROPOUT_PARAMETER}" \
    --use-finetuned-model \
    --task=finetuning \
    $DRY_RUN_OPTION \
    $LOCAL_OPTION \
    --run-only-selected-configs-option "${RUN_ONLY_SELECTED_CONFIGS_OPTION}" # & # Uncomment the ampersand ('&') to run in background
done



# ================================================================== #
# Exit submission script
echo ">>> Submission script finished."
echo ">>> Exiting ..."
exit $?
# ================================================================== #
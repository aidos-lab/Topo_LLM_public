#!/bin/bash

# ================================================================== #
# Start submission script
echo ">>> Submission script started."
# ================================================================== #

# # # #
#
# SUBMISSION_MODE="local"
SUBMISSION_MODE="hpc_submission"

# >>>> BEGIN: Select the run option
# >>>> Use the following to dry run with all configurations
#
RUN_OPTION="dry_run"
RUN_ONLY_SELECTED_CONFIGS_OPTION="run_all"
#
# >>>> Use the following to dry run a random configuration
#
# RUN_OPTION="dry_run"
# RUN_ONLY_SELECTED_CONFIGS_OPTION="run_single_random"
#
# >>> Use the following to run a random configuration
#
# RUN_OPTION="do_submission"
# RUN_ONLY_SELECTED_CONFIGS_OPTION="run_single_random"
#
# >>>> Use the following to run with all configurations
#
# RUN_OPTION="do_submission"
# RUN_ONLY_SELECTED_CONFIGS_OPTION="run_all"
#
# >>>> END: Select the run option


DROPOUT_PARAMETER_LIST=(
  "0.05"
  # "0.06"
  # "0.07"
  # "0.15"
  # "0.2"
  # "0.25"
  # "0.3"
  # "0.35"
)

# SKIP_FINETUNING_OPTION="--additional-overrides feature_flags.finetuning.skip_finetuning=true"
# USE_WANDB_FALSE_OPTION="--additional-overrides feature_flags.wandb.use_wandb=false"

# The "mps" backend does not support FP16 training,
# this option allows us to deactivate it when runnign locally
if [ "$SUBMISSION_MODE" == "local" ]; then
  FP16="false"
else
  # Set this to true for running on the cluster
  FP16="true"
fi

echo ">>> FP16: ${FP16}"

### Finetuning
#

for DROPOUT_PARAMETER in "${DROPOUT_PARAMETER_LIST[@]}"; do
  # Notes:
  # - For finetuning, the following arguments are just placeholders which have no effect on the actual finetuning process:
  #   --experiment-selector
  #   --experiment-stage
  #
  # - Do not place quotation marks around the variables in the arguments which can be empty
  #   (e.g., $SKIP_FINETUNING_OPTION, $USE_WANDB_FALSE_OPTION)
  poetry run submit_jobs \
    --experiment-selector "multiwoz21_different_data_subsampling_number_of_samples" \
    --experiment-stage "skip_compute_embeddings_but_do_multiple_pipeline_runs" \
    --finetuning-datasets-list-option "multiwoz21_and_reddit_small" \
    --wandb-project "Topo_LLM_finetuning_from_submission_script_dropout_different_choices_with_parameter_and_gradient_logging" \
    --additional-overrides "+finetuning.base_model.dropout.mode=modify_roberta_dropout_parameters" \
    --additional-overrides "+finetuning.base_model.dropout.probabilities.hidden_dropout_prob=${DROPOUT_PARAMETER}" \
    --additional-overrides "+finetuning.base_model.dropout.probabilities.attention_probs_dropout_prob=${DROPOUT_PARAMETER}" \
    --fp16 "${FP16}" \
    $SKIP_FINETUNING_OPTION \
    $USE_WANDB_FALSE_OPTION \
    --use-finetuned-model \
    --task=finetuning \
    --submission-mode "${SUBMISSION_MODE}" \
    --run-option "${RUN_OPTION}" \
    --run-only-selected-configs-option "${RUN_ONLY_SELECTED_CONFIGS_OPTION}" # & # Uncomment the ampersand ('&') to run in background
done



# ================================================================== #
# Exit submission script
echo ">>> Submission script finished."
echo ">>> Exiting ..."
exit $?
# ================================================================== #
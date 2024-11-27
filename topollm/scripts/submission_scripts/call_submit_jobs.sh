#!/bin/bash

# ================================================================== #
# Start submission script
echo ">>> Submission script started."
# ================================================================== #

### Finetuning
#
# poetry run submit_jobs \
#   --experiment-selector multiwoz21_different_data_subsampling_number_of_samples \
#   --experiment-stage skip_compute_embeddings_and_multiple_pipeline_runs \
#   --use-finetuned-model \
#   --task=finetuning \
#   --dry-run

# poetry run submit_jobs \
#   --experiment-selector multiwoz21_different_data_subsampling_number_of_samples \
#   --experiment-stage compute_embeddings_plus_single_pipeline_run \
#   --use-roberta-base \
#   --task=pipeline \
#   --dry-run

# poetry run submit_jobs \
#   --experiment-selector multiwoz21_different_data_subsampling_number_of_samples \
#   --experiment-stage skip_compute_embeddings_and_multiple_pipeline_runs \
#   --use-roberta-base \
#   --task=pipeline \
#   --dry-run

# ++++ Experiment > High checkpoint resolution ++++

# Leave DRY_RUN_OPTION empty to run without the dry-run option, i.e., to actually submit the jobs
#
# DRY_RUN_OPTION="--dry-run"

RUN_ONLY_SELECTED_CONFIGS_OPTION="run_all"
# RUN_ONLY_SELECTED_CONFIGS_OPTION="run_single_random"

DATA_LIST_OPTION_LIST=(
  "reddit_only"
  "multiwoz21_only"
)

for DATA_LIST_OPTION in "${DATA_LIST_OPTION_LIST[@]}"; do
  poetry run submit_jobs \
    --data-list-option "${DATA_LIST_OPTION}" \
    --experiment-selector fixed_parameters_high_checkpoint_resolution \
    --experiment-stage compute_embeddings_plus_single_pipeline_run \
    --use-finetuned-model \
    --task=pipeline \
    $DRY_RUN_OPTION \
    --run-only-selected-configs-option "${RUN_ONLY_SELECTED_CONFIGS_OPTION}" & # Uncomment the ampersand ('&') to run in background
done

# ================================================================== #
# Exit submission script
echo ">>> Submission script finished."
echo ">>> Exiting ..."
exit $?
# ================================================================== #
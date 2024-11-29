#!/bin/bash

# ================================================================== #
# Start submission script
echo ">>> Submission script started."
# ================================================================== #

# Initialize dry_run flag to false
dry_run=false

# Parse command line options
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --dry_run)
      dry_run=true
      shift # Remove the --dry_run argument
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Leave DRY_RUN_OPTION empty to run without the dry-run option, i.e., to actually submit the jobs
if [ "$dry_run" = true ]; then
  DRY_RUN_OPTION="--dry-run"
else
  DRY_RUN_OPTION=""
fi

# poetry run submit_jobs \
#   --experiment-selector multiwoz21_different_data_subsampling_number_of_samples \
#   --experiment-stage compute_embeddings_plus_single_pipeline_run \
#   --use-roberta-base \
#   --task=pipeline \
#   $DRY_RUN_OPTION

# poetry run submit_jobs \
#   --experiment-selector multiwoz21_different_data_subsampling_number_of_samples \
#   --experiment-stage skip_compute_embeddings_and_multiple_pipeline_runs \
#   --use-roberta-base \
#   --task=pipeline \
#   $DRY_RUN_OPTION


RUN_ONLY_SELECTED_CONFIGS_OPTION="run_all"
# RUN_ONLY_SELECTED_CONFIGS_OPTION="run_single_random"
# RUN_ONLY_SELECTED_CONFIGS_OPTION="run_only_first"

DATA_LIST_OPTION_LIST=(
  "reddit_only"
  "multiwoz21_only"
)

# ++++ Experiment > High checkpoint resolution ++++
#
# EXPERIMENT_SELECTOR="fixed_parameters_high_checkpoint_resolution"

# ++++ Experiment > Dropout analysis ++++
#
# EXPERIMENT_SELECTOR="exploratory_dropout_analysis_coarse_checkpoint_resolution"
EXPERIMENT_SELECTOR="tiny_dropout_variations_coarse_checkpoint_resolution"

for DATA_LIST_OPTION in "${DATA_LIST_OPTION_LIST[@]}"; do
  poetry run submit_jobs \
    --data-list-option "${DATA_LIST_OPTION}" \
    --experiment-selector "${EXPERIMENT_SELECTOR}" \
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
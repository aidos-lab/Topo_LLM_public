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

poetry run submit_jobs \
  --experiment-selector fixed_parameters_reddit_dataset_high_checkpoint_resolution \
  --experiment-stage compute_embeddings_plus_single_pipeline_run \
  --use-finetuned-model \
  --task=pipeline \
  --run-only-first-config \
  --dry-run


# ================================================================== #
# Exit submission script
echo ">>> Submission script finished."
echo ">>> Exiting ..."
exit $?
# ================================================================== #
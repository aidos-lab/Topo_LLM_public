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
# RUN_OPTION="dry_run"
# RUN_ONLY_SELECTED_CONFIGS_OPTION="run_all"
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
RUN_OPTION="do_submission"
RUN_ONLY_SELECTED_CONFIGS_OPTION="run_all"
#
# >>>> END: Select the run option

FINETUNING_DATASETS_LIST_OPTION_LIST=(
  # "iclr_small"
  "multiwoz21_small",
  "reddit_small",
  # "sgd_small"
  # "wikitext_small"
  # "multiwoz21_and_reddit_small"
)

# Example of a list of parameters to loop through
PARAMETER_LIST=(
  "PLACEHOLDER_0"
  # "PLACEHOLDER_1"
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

for FINETUNING_DATASETS_LIST_OPTION in "${FINETUNING_DATASETS_LIST_OPTION_LIST[@]}"; do
  echo ">>> FINETUNING_DATASETS_LIST_OPTION: ${FINETUNING_DATASETS_LIST_OPTION}"

  for PARAMETER in "${PARAMETER_LIST[@]}"; do
    echo ">>> PARAMETER: ${PARAMETER}"

    # Notes:
    # - For finetuning, the following arguments are just placeholders which have no effect on the actual finetuning process:
    #   --experiment-selector
    #   --experiment-stage
    #
    # - Do not place quotation marks around the variables in the arguments which can be empty
    #   (e.g., $SKIP_FINETUNING_OPTION, $USE_WANDB_FALSE_OPTION)
    poetry run submit_jobs \
      --experiment-selector "sensitivity_analysis_multiwoz21_different_data_subsampling_number_of_samples" \
      --experiment-stage "skip_compute_embeddings_but_do_multiple_pipeline_runs" \
      --finetuning-base-model-list-option "gpt2_medium" \
      --finetuning-datasets-list-option "${FINETUNING_DATASETS_LIST_OPTION}" \
      --wandb-project "Topo_LLM_gpt2_finetuning_from_submission_script_for_5_epochs_and_linear_lr_schedule" \
      --fp16 "${FP16}" \
      --model-group-option "roberta_base_finetuned_for_few_epochs_old_and_new_data_single_seed_last_checkpoint" \
      $SKIP_FINETUNING_OPTION \
      $USE_WANDB_FALSE_OPTION \
      --task=finetuning \
      --submission-mode "${SUBMISSION_MODE}" \
      --run-option "${RUN_OPTION}" \
      --run-only-selected-configs-option "${RUN_ONLY_SELECTED_CONFIGS_OPTION}" # & # Uncomment the ampersand ('&') to run in background
  done
  
done

# TODO
# TODO: Find out why the following error occurs when running the script
#
# >>> FINETUNING_DATASETS_LIST_OPTION: multiwoz21_small,
# >>> PARAMETER: PLACEHOLDER_0
# Warning: In a future version of Poetry, PyPI will be disabled automatically if at least one custom primary source is configured. In order to avoid a breaking change and make your pyproject.toml forward compatible, add PyPI explicitly via 'poetry source add pypi'. By the way, this has the advantage that you can set the priority of PyPI as with any other source.
# Usage: submit_jobs [OPTIONS]
# Try 'submit_jobs --help' for help.

# Error: Invalid value for '--finetuning-datasets-list-option': multiwoz21_small,
# >>> FINETUNING_DATASETS_LIST_OPTION: reddit_small,
# >>> PARAMETER: PLACEHOLDER_0
# Warning: In a future version of Poetry, PyPI will be disabled automatically if at least one custom primary source is configured. In order to avoid a breaking change and make your pyproject.toml forward compatible, add PyPI explicitly via 'poetry source add pypi'. By the way, this has the advantage that you can set the priority of PyPI as with any other source.
# Usage: submit_jobs [OPTIONS]
# Try 'submit_jobs --help' for help.

# Error: Invalid value for '--finetuning-datasets-list-option': reddit_small,
# >>> Submission script finished.
# >>> Exiting ...



# ================================================================== #
# Exit submission script
echo ">>> Submission script finished."
echo ">>> Exiting ..."
exit $?
# ================================================================== #
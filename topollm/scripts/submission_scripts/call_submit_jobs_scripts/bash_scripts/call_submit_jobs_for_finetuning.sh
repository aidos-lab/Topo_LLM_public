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

# SUBMISSION_MODE_SELECTOR="ALL_DATASETS"
SUBMISSION_MODE_SELECTOR="WIKITEXT_SMALL_DATA_MULTIPLE_DATA_SUBSAMPLING_SAMPLING_SEEDS"

# Select options based on the submission mode
if [ "$SUBMISSION_MODE_SELECTOR" == "ALL_DATASETS" ]; then
    FINETUNING_DATASETS_LIST_OPTION_LIST=(
        "iclr_small"
        "multiwoz21_small"
        "reddit_small"
        "sgd_small"
        "wikitext_small"
    )
elif [ "$SUBMISSION_MODE_SELECTOR" == "WIKITEXT_SMALL_DATA_MULTIPLE_DATA_SUBSAMPLING_SAMPLING_SEEDS" ]; then
    FINETUNING_DATASETS_LIST_OPTION_LIST=(
        "wikitext_small"
    )

    FINETUNING_TRAIN_DATASET_DATA_SUBSAMPLING_SAMPLING_MODE_OPTION="--additional-overrides finetuning.finetuning_datasets.train_dataset.data_subsampling.sampling_mode=random"
    FINETUNING_TRAIN_DATASET_DATA_SUBSAMPLING_SAMPLING_SEED_OPTION="--additional-overrides finetuning.finetuning_datasets.train_dataset.data_subsampling.sampling_seed=112,113,114"

    WANDB_OPTION="Topo_LLM_roberta-base_finetuning_for_ep-5_lr-linear_no_freeze_different_data_subsampling_seeds"
else
  echo ">>> Unknown submission mode selector: SUBMISSION_MODE_SELECTOR=${SUBMISSION_MODE_SELECTOR}"
  echo ">>> Will not set any additional options."
fi

# # # #

FINETUNING_BASE_MODEL_LIST_OPTION="roberta_base"
# FINETUNING_BASE_MODEL_LIST_OPTION="gpt2_medium"

# WANDB_OPTION="Topo_LLM_roberta-base_finetuning_from_submission_script_for_5_epochs_and_linear_lr_schedule_no_freeze"
# WANDB_OPTION="Topo_LLM_roberta-base_finetuning_from_submission_script_for_5_epochs_and_linear_lr_schedule_freeze_lm_head"
# WANDB_OPTION="Topo_LLM_gpt2_finetuning_from_submission_script_for_5_epochs_and_linear_lr_schedule_no_freeze"

# Note: The wandb project name must not exceeded 128 characters.
#
# Truncate the WANDB_OPTION to 128 characters
WANDB_OPTION="${WANDB_OPTION:0:128}"

# Example of a list of parameters to loop through
PARAMETER_LIST=(
  "PLACEHOLDER_0"
  # "PLACEHOLDER_1"
)

# # # # # # # #
# Note:
# If you do not want to set the following options, you can leave the variables as empty strings,
# or you can comment them out.

# SKIP_FINETUNING_OPTION="--additional-overrides feature_flags.finetuning.skip_finetuning=true"
# USE_WANDB_FALSE_OPTION="--additional-overrides feature_flags.wandb.use_wandb=false"

FINETUNING_SAVE_STEPS_OPTION="--additional-overrides finetuning.save_steps=100" # <-- Note: This will lead to a large number of checkpoints being saved

FINETUNING_GRADIENT_MODIFIER_OPTION="--additional-overrides finetuning/gradient_modifier=do_nothing"
# FINETUNING_GRADIENT_MODIFIER_OPTION="--additional-overrides finetuning/gradient_modifier=freeze_lm_head_bert-style-models"
# FINETUNING_GRADIENT_MODIFIER_OPTION="--additional-overrides finetuning/gradient_modifier=freeze_lm_head_and_word_embeddings_bert-style-models"


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
      --model-group-option "roberta_base_finetuned_for_few_epochs_old_and_new_data_single_seed_last_checkpoint" \
      --finetuning-base-model-list-option "${FINETUNING_BASE_MODEL_LIST_OPTION}" \
      --finetuning-datasets-list-option "${FINETUNING_DATASETS_LIST_OPTION}" \
      --wandb-project "${WANDB_OPTION}" \
      --fp16 "${FP16}" \
      $SKIP_FINETUNING_OPTION \
      $USE_WANDB_FALSE_OPTION \
      $FINETUNING_TRAIN_DATASET_DATA_SUBSAMPLING_SAMPLING_MODE_OPTION \
      $FINETUNING_TRAIN_DATASET_DATA_SUBSAMPLING_SAMPLING_SEED_OPTION \
      $FINETUNING_SAVE_STEPS_OPTION \
      $FINETUNING_GRADIENT_MODIFIER_OPTION \
      --task=finetuning \
      --submission-mode "${SUBMISSION_MODE}" \
      --run-option "${RUN_OPTION}" \
      --run-only-selected-configs-option "${RUN_ONLY_SELECTED_CONFIGS_OPTION}" # & # Uncomment the ampersand ('&') to run in background
  done
  
done

# ================================================================== #
# Exit submission script
echo ">>> Submission script finished."
echo ">>> Exiting ..."
exit $?
# ================================================================== #
#!/bin/bash

# ================================================================== #

# Default values
SUBMISSION_MODE="hpc_submission"
DRY_RUN_FLAG=""

# Note: 
# 16GB of memory is not enough for the embeddings data prep step
# on the multiwoz21_train and reddit_train datasets.
MEMORY="32"

# Flags to select tasks
DO_PIPELINE="false"
DO_PERPLEXITY="false"
DO_LOCAL_ESTIMATES_COMPUTATION="false"
DO_FINETUNING="false"

# Flags to select base or fine-tuned model
USE_ROBERTA_BASE_MODEL="false"
USE_FINETUNED_MODEL="false"

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --local)
      SUBMISSION_MODE="local"
      shift # Remove --local from processing
      ;;
    --dry_run)
      DRY_RUN_FLAG="--dry_run"
      shift # Remove --dry_run from processing
      ;;
    --do_pipeline)
      DO_PIPELINE="true"
      shift # Remove --do_pipeline from processing
      ;;
    --do_perplexity)
      DO_PERPLEXITY="true"
      shift # Remove --do_perplexity from processing
      ;;
    --do_finetuning)
      DO_FINETUNING="true"
      shift # Remove --do_finetuning from processing
      ;;
    --do_local_estimates_computation)
      DO_LOCAL_ESTIMATES_COMPUTATION="true"
      shift # Remove --do_local_estimates_computation from processing
      ;;
    --use_roberta_base_model)
      USE_ROBERTA_BASE_MODEL="true"
      shift # Remove --use_roberta_base_model from processing
      ;;
    --use_finetuned_model)
      USE_FINETUNED_MODEL="true"
      shift # Remove --use_finetuned_model from processing
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Print a message of no task is selected
if [ "$DO_PIPELINE" = "false" ] && [ "$DO_PERPLEXITY" = "false" ] && [ "$DO_LOCAL_ESTIMATES_COMPUTATION" = "false" ] && [ "$DO_FINETUNING" = "false" ]; then
  echo ">>> NOTE: No task is selected. This script will not submit any jobs."
  echo ">>> NOTE: Use at least one of the following options:"
  echo ">>> NOTE:   --do_pipeline"
  echo ">>> NOTE:   --do_perplexity"
  echo ">>> NOTE:   --do_local_estimates_computation"
  echo ">>> NOTE:   --do_finetuning"
fi

# Warn if neither model options are set
if [ "$USE_ROBERTA_BASE_MODEL" = "false" ] && [ "$USE_FINETUNED_MODEL" = "false" ]; then
  echo ">>> WARNING: No model selected. Please specify --use_roberta_base_model or --use_finetuned_model."
  exit 1
fi
# Warn if both model options are set
if [ "$USE_ROBERTA_BASE_MODEL" = "true" ] && [ "$USE_FINETUNED_MODEL" = "true" ]; then
  echo ">>> WARNING: Both base model and fine-tuned model selected. Please select only one."
  exit 1
fi

# ================================================================== #


# ================================================================== #

# DATA_LIST="full"
DATA_LIST="multiwoz21_and_reddit"
# DATA_LIST="multiwoz21_train_and_reddit_train"
# DATA_LIST="multiwoz21_only"
# DATA_LIST="reddit_only"

# DATA_LIST="only_train"
# LANGUAGE_MODEL_LIST="selected_finetuned_few_epochs_from_roberta_base"
# CHECKPOINT_NO_LIST="selected"

# DATA_LIST="debug"
# LANGUAGE_MODEL_LIST="only_roberta_base"

# DATA_NUMBER_OF_SAMPLES_LIST_OPTION="none"
# DATA_NUMBER_OF_SAMPLES_LIST_OPTION="fixed_3000"
DATA_NUMBER_OF_SAMPLES_LIST_OPTION="fixed_10000"
# DATA_NUMBER_OF_SAMPLES_LIST_OPTION="up_to_10000_with_step_size_2000"

# DATA_SUBSAMPLING_SAMPLING_SEED_LIST_OPTION="default"
# DATA_SUBSAMPLING_SAMPLING_SEED_LIST_OPTION="fixed_777"
DATA_SUBSAMPLING_SAMPLING_SEED_LIST_OPTION="ten_seeds"

# ================================================================== #

# Uncomment the following to skip the compute_and_store_embeddings step:
#
# SKIP_COMPUTE_AND_STORE_EMBEDDINGS="--skip_compute_and_store_embeddings"

EMBEDDINGS_DATA_PREP_SAMPLING_MODE="random"
# EMBEDDINGS_DATA_PREP_SAMPLING_MODE="take_first"

# EMBEDDINGS_DATA_PREP_SAMPLING_SEED_LIST_OPTION="default"
# EMBEDDINGS_DATA_PREP_SAMPLING_SEED_LIST_OPTION="two_seeds"
EMBEDDINGS_DATA_PREP_SAMPLING_SEED_LIST_OPTION="five_seeds"
# EMBEDDINGS_DATA_PREP_SAMPLING_SEED_LIST_OPTION="ten_seeds"
# EMBEDDINGS_DATA_PREP_SAMPLING_SEED_LIST_OPTION="twenty_seeds"

# EMBEDDINGS_DATA_PREP_NUM_SAMPLES_LIST_OPTION="default"
# EMBEDDINGS_DATA_PREP_NUM_SAMPLES_LIST_OPTION="single_choice_50000"
EMBEDDINGS_DATA_PREP_NUM_SAMPLES_LIST_OPTION="single_choice_100000"
# EMBEDDINGS_DATA_PREP_NUM_SAMPLES_LIST_OPTION="five_choices_10000_steps"
# EMBEDDINGS_DATA_PREP_NUM_SAMPLES_LIST_OPTION="single_choice_250000"

LOCAL_ESTIMATES_FILTERING_NUM_SAMPLES_LIST="default"
# LOCAL_ESTIMATES_FILTERING_NUM_SAMPLES_LIST="few_small_steps_num_samples"
# LOCAL_ESTIMATES_FILTERING_NUM_SAMPLES_LIST="up_to_90000_with_step_size_5000_num_samples"
# LOCAL_ESTIMATES_FILTERING_NUM_SAMPLES_LIST="up_to_90000_with_step_size_10000_num_samples"
# LOCAL_ESTIMATES_FILTERING_NUM_SAMPLES_LIST="up_to_100000_with_step_size_20000_num_samples"

# LOCAL_ESTIMATES_POINTWISE_ABSOLUTE_N_NEIGHBORS_LIST="powers_of_two_up_to_1024"
LOCAL_ESTIMATES_POINTWISE_ABSOLUTE_N_NEIGHBORS_LIST="single_choice_128"

# Configure model settings based on selected model options
if [ "$USE_ROBERTA_BASE_MODEL" = "true" ]; then
  ####################################
  ### With POS tags for base model ###
  LANGUAGE_MODEL_LIST="only_roberta_base"
  LANGUAGE_MODEL_SEED_LIST="do_not_set"
  CHECKPOINT_NO_LIST="selected" # Ignored for the base model
  FINETUNING_REGIME="few_epochs" # Ignored for the base model
fi

if [ "$USE_FINETUNED_MODEL" = "true" ]; then
  ################################################################
  ### With POS tags for finetuned models and three checkpoints ###
  LANGUAGE_MODEL_LIST="selected_finetuned_many_epochs_from_roberta_base"
  LANGUAGE_MODEL_SEED_LIST="one_seed"
  CHECKPOINT_NO_LIST="only_beginning_and_middle_and_end"
  FINETUNING_REGIME="many_epochs_with_overfitting_risk"
fi

ADD_PREFIX_SPACE_FLAG="--add_prefix_space"
CREATE_POS_TAGS_FLAG="--create_pos_tags"

# ================================================================== #

# ================================================================== #
if [ "$DO_PIPELINE" = "true" ]; then
  echo ">>> Submitting pipeline jobs ..."
  # Note: To avoid re-running the expensive steps
  # - compute_embeddings and
  # - embeddings_data_prep,
  # we do not use the 'LocalEstimatesFilteringNumSamplesOption' in the pipeline.
  poetry run submit_jobs \
      --task="pipeline" \
      --queue="DSML" \
      --template="DSML" \
      --memory=$MEMORY \
      --data_list=$DATA_LIST \
      --data_subsampling_number_of_samples_list_option=$DATA_NUMBER_OF_SAMPLES_LIST_OPTION \
      --data_subsampling_sampling_seed_list_option=$DATA_SUBSAMPLING_SAMPLING_SEED_LIST_OPTION \
      $CREATE_POS_TAGS_FLAG \
      --language_model_list=$LANGUAGE_MODEL_LIST \
      --checkpoint_no_list=$CHECKPOINT_NO_LIST \
      --language_model_seed_list=$LANGUAGE_MODEL_SEED_LIST \
      $ADD_PREFIX_SPACE_FLAG \
      --finetuning_regime=$FINETUNING_REGIME \
      --embeddings_data_prep_sampling_mode=$EMBEDDINGS_DATA_PREP_SAMPLING_MODE \
      --embeddings_data_prep_sampling_seed_list_option=$EMBEDDINGS_DATA_PREP_SAMPLING_SEED_LIST_OPTION \
      --embeddings_data_prep_num_samples_list_option=$EMBEDDINGS_DATA_PREP_NUM_SAMPLES_LIST_OPTION \
      $SKIP_COMPUTE_AND_STORE_EMBEDDINGS \
      --submission_mode=$SUBMISSION_MODE \
      $DRY_RUN_FLAG
  echo ">>> Submitting pipeline jobs DONE"
fi
# ================================================================== #
       
# ================================================================== #
if [ "$DO_LOCAL_ESTIMATES_COMPUTATION" = "true" ]; then
  echo ">>> Submitting local estimates computation jobs ..."
  # This is a CPU task, so we do not ask for a GPU.
  poetry run submit_jobs \
      --task="local_estimates_computation" \
      --template="CPU" \
      --queue="DEFAULT" \
      --ngpus="0" \
      --walltime="08:00:00" \
      --data_list=$DATA_LIST \
      --data_subsampling_number_of_samples_list_option=$DATA_NUMBER_OF_SAMPLES_LIST_OPTION \
      --data_subsampling_sampling_seed_list_option=$DATA_SUBSAMPLING_SAMPLING_SEED_LIST_OPTION \
      $CREATE_POS_TAGS_FLAG \
      --language_model_list=$LANGUAGE_MODEL_LIST \
      --checkpoint_no_list=$CHECKPOINT_NO_LIST \
      --language_model_seed_list=$LANGUAGE_MODEL_SEED_LIST \
      $ADD_PREFIX_SPACE_FLAG \
      --finetuning_regime=$FINETUNING_REGIME \
      --embeddings_data_prep_sampling_mode=$EMBEDDINGS_DATA_PREP_SAMPLING_MODE \
      --embeddings_data_prep_sampling_seed_list_option=$EMBEDDINGS_DATA_PREP_SAMPLING_SEED_LIST_OPTION \
      --embeddings_data_prep_num_samples_list_option=$EMBEDDINGS_DATA_PREP_NUM_SAMPLES_LIST_OPTION \
      --local_estimates_filtering_num_samples_list=$LOCAL_ESTIMATES_FILTERING_NUM_SAMPLES_LIST \
      --local_estimates_pointwise_absolute_n_neighbors_list=$LOCAL_ESTIMATES_POINTWISE_ABSOLUTE_N_NEIGHBORS_LIST \
      --submission_mode=$SUBMISSION_MODE \
      $DRY_RUN_FLAG
  echo ">>> Submitting local estimates computation jobs DONE"
fi
# ================================================================== #

# ================================================================== #
if [ "$DO_PERPLEXITY" = "true" ]; then
  echo ">>> Submitting perplexity jobs ..."
  # Note: Do not use the CREATE_POS_TAGS_FLAG for the perplexity task.
  poetry run submit_jobs \
      --task="perplexity" \
      --queue="CUDA" \
      --template="RTX6000" \
      --data_list=$DATA_LIST \
      --data_subsampling_number_of_samples_list_option=$DATA_NUMBER_OF_SAMPLES_LIST_OPTION \
      --data_subsampling_sampling_seed_list_option=$DATA_SUBSAMPLING_SAMPLING_SEED_LIST_OPTION \
      --language_model_list=$LANGUAGE_MODEL_LIST \
      --checkpoint_no_list=$CHECKPOINT_NO_LIST \
      --language_model_seed_list=$LANGUAGE_MODEL_SEED_LIST \
      $ADD_PREFIX_SPACE_FLAG \
      --finetuning_regime=$FINETUNING_REGIME \
      --submission_mode=$SUBMISSION_MODE \
      $DRY_RUN_FLAG
  echo ">>> Submitting perplexity jobs DONE"
fi
# ================================================================== #

# ================================================================== #
if [ "$DO_FINETUNING" = "true" ]; then
  echo ">>> Submitting finetuning jobs ..."
  poetry run submit_jobs \
    --task="finetuning" \
    --queue="CUDA" \
    --template="RTX6000" \
    --finetuning_datasets_list="manual_in_python_script" \
    --finetuning_seed_list="one_seed" \
    --finetuning_regime="many_epochs_with_overfitting_risk" \
    --submission_mode=$SUBMISSION_MODE \
    --wandb_project="Topo_LLM_finetuning_from_submission_script_DEBUG" \
    $DRY_RUN_FLAG
  echo ">>> Submitting finetuning jobs ..."
fi

# Exit submission script
echo ">>> Submission script finished."
echo ">>> Exiting ..."
exit $?
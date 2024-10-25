#!/bin/bash

# ================================================================== #

# Default values
SUBMISSION_MODE="hpc_submission"
DRY_RUN_FLAG=""

DO_PIPELINE="false"
DO_PERPLEXITY="false"
DO_LOCAL_ESTIMATES_COMPUTATION="false"


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
    --do_local_estimates_computation)
      DO_LOCAL_ESTIMATES_COMPUTATION="true"
      shift # Remove --do_local_estimates_computation from processing
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Print a message of no task is selected
if [ "$DO_PIPELINE" = "false" ] && [ "$DO_PERPLEXITY" = "false" ] && [ "$DO_LOCAL_ESTIMATES_COMPUTATION" = "false" ]; then
  echo ">>> NOTE: No task is selected. This script will not submit any jobs."
  echo ">>> NOTE: Use at least one of the following options:"
  echo ">>> NOTE:   --do_pipeline"
  echo ">>> NOTE:   --do_perplexity"
  echo ">>> NOTE:   --do_local_estimates_computation"
fi

# ================================================================== #


# ================================================================== #
# Debug setup

# DATA_LIST="only_train"
# LANGUAGE_MODEL_LIST="selected_finetuned_few_epochs_from_roberta_base"
# CHECKPOINT_NO_LIST="selected"

# DATA_LIST="debug"
# LANGUAGE_MODEL_LIST="only_roberta_base"

# ================================================================== #

SKIP_COMPUTE_AND_STORE_EMBEDDINGS="--skip_compute_and_store_embeddings"

EMBEDDINGS_DATA_PREP_SAMPLING_MODE="random"
# EMBEDDINGS_DATA_PREP_SAMPLING_MODE="take_first"

# EMBEDDINGS_DATA_PREP_SAMPLING_SEED_LIST_OPTION="default"
# EMBEDDINGS_DATA_PREP_SAMPLING_SEED_LIST_OPTION="two_seeds"
EMBEDDINGS_DATA_PREP_SAMPLING_SEED_LIST_OPTION="five_seeds"
# EMBEDDINGS_DATA_PREP_SAMPLING_SEED_LIST_OPTION="ten_seeds"
# EMBEDDINGS_DATA_PREP_SAMPLING_SEED_LIST_OPTION="twenty_seeds"

# EMBEDDINGS_DATA_PREP_NUM_SAMPLES_LIST="default"
EMBEDDINGS_DATA_PREP_NUM_SAMPLES_LIST="five_choices_10000_steps"

# LOCAL_ESTIMATES_FILTERING_NUM_SAMPLES_LIST="few_small_steps_num_samples"
LOCAL_ESTIMATES_FILTERING_NUM_SAMPLES_LIST="medium_small_steps_num_samples"
# LOCAL_ESTIMATES_FILTERING_NUM_SAMPLES_LIST="many_small_steps_num_samples"
# LOCAL_ESTIMATES_FILTERING_NUM_SAMPLES_LIST="many_large_steps_num_samples"

LOCAL_ESTIMATES_POINTWISE_ABSOLUTE_N_NEIGHBORS_LIST="powers_of_two_up_to_1024"


####################################
### With POS tags for base model ###
#
# DATA_LIST="full"
DATA_LIST="multiwoz21_and_reddit"

LANGUAGE_MODEL_LIST="only_roberta_base"
LANGUAGE_MODEL_SEED_LIST="do_not_set"
CHECKPOINT_NO_LIST="selected" # Will be ignored for the base model
FINETUNING_REGIME="few_epochs" # Will be ignored for the base model
ADD_PREFIX_SPACE_FLAG="--add_prefix_space"
CREATE_POS_TAGS_FLAG="--create_pos_tags"

##############################################################################
### With POS tags for finetuned models and three checkpoints and two seeds ###
#
# DATA_LIST="full"
# DATA_LIST="multiwoz21_and_reddit"

# LANGUAGE_MODEL_LIST="selected_finetuned_many_epochs_from_roberta_base"
# LANGUAGE_MODEL_SEED_LIST="one_seed"
# CHECKPOINT_NO_LIST="only_beginning_and_middle_and_end"
# FINETUNING_REGIME="many_epochs_with_overfitting_risk"
# ADD_PREFIX_SPACE_FLAG="--add_prefix_space"
# CREATE_POS_TAGS_FLAG="--create_pos_tags"

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
      --data_list=$DATA_LIST \
      $CREATE_POS_TAGS_FLAG \
      --language_model_list=$LANGUAGE_MODEL_LIST \
      --checkpoint_no_list=$CHECKPOINT_NO_LIST \
      --language_model_seed_list=$LANGUAGE_MODEL_SEED_LIST \
      $ADD_PREFIX_SPACE_FLAG \
      --finetuning_regime=$FINETUNING_REGIME \
      --embeddings_data_prep_sampling_mode=$EMBEDDINGS_DATA_PREP_SAMPLING_MODE \
      --embeddings_data_prep_sampling_seed_list_option=$EMBEDDINGS_DATA_PREP_SAMPLING_SEED_LIST_OPTION \
      --embeddings_data_prep_num_samples_list=$EMBEDDINGS_DATA_PREP_NUM_SAMPLES_LIST \
      $SKIP_COMPUTE_AND_STORE_EMBEDDINGS \
      --submission_mode=$SUBMISSION_MODE \
      $DRY_RUN_FLAG
  echo ">>> Submitting pipeline jobs DONE"
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
      $CREATE_POS_TAGS_FLAG \
      --language_model_list=$LANGUAGE_MODEL_LIST \
      --checkpoint_no_list=$CHECKPOINT_NO_LIST \
      --language_model_seed_list=$LANGUAGE_MODEL_SEED_LIST \
      $ADD_PREFIX_SPACE_FLAG \
      --finetuning_regime=$FINETUNING_REGIME \
      --embeddings_data_prep_sampling_mode=$EMBEDDINGS_DATA_PREP_SAMPLING_MODE \
      --embeddings_data_prep_sampling_seed_list_option=$EMBEDDINGS_DATA_PREP_SAMPLING_SEED_LIST_OPTION \
      --embeddings_data_prep_num_samples_list=$EMBEDDINGS_DATA_PREP_NUM_SAMPLES_LIST \
      --local_estimates_filtering_num_samples_list=$LOCAL_ESTIMATES_FILTERING_NUM_SAMPLES_LIST \
      --local_estimates_pointwise_absolute_n_neighbors_list=$LOCAL_ESTIMATES_POINTWISE_ABSOLUTE_N_NEIGHBORS_LIST \
      --submission_mode=$SUBMISSION_MODE \
      $DRY_RUN_FLAG
  echo ">>> Submitting local estimates computation jobs DONE"
fi
# ================================================================== #

# Exit submission script
echo ">>> Submission script finished."
echo ">>> Exiting ..."
exit 0
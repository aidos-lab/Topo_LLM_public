#!/bin/bash

# ================================================================== #

# Default values
SUBMISSION_MODE="hpc_submission"
DRY_RUN_FLAG=""
RUN_ONLY_FIRST_CONFIG_OPTION_FLAG=""

# Note: 
# - 16GB of memory is not enough for the embeddings data prep step
#   for dataset subsampling sample size 10_000 on the multiwoz21_train and reddit_train datasets.
# - 32GB of memory is enough for the embeddings data prep step
#   for dataset subsampling sample size 12_000 on the multiwoz21_train and reddit_train datasets.

MEMORY="32"
NCPUS="2"
NGPUS="1"

TEMPLATE="DSML"
QUEUE="DSML"

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
    --run_only_first_config_option)
      RUN_ONLY_FIRST_CONFIG_OPTION_FLAG="--run_only_first_config_option"
      shift # Remove --run_only_first_config_option from processing
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
  FINETUNING_REGIME="many_epochs_with_overfitting_risk"
  
  # LANGUAGE_MODEL_SEED_LIST="one_seed"
  # LANGUAGE_MODEL_SEED_LIST="fixed_seeds_1235_1236"
  LANGUAGE_MODEL_SEED_LIST="fixed_seeds_1234_1235_1236"
  
  CHECKPOINT_NO_LIST="only_beginning_and_middle_and_end"
fi

ADD_PREFIX_SPACE_FLAG="--add_prefix_space"
CREATE_POS_TAGS_FLAG="--create_pos_tags"


# ================================================================== #


# ================================================================== #
# Note that these values might be overridden by the individual experiment setup below.

# DATA_LIST="full"
# DATA_LIST="multiwoz21_and_reddit"
# DATA_LIST="multiwoz21_train_and_reddit_train"
DATA_LIST="multiwoz21_only"
# DATA_LIST="reddit_only"
# DATA_LIST="only_train"
# DATA_LIST="debug"

DATA_SUBSAMPLING_SAMPLING_SEED_LIST_OPTION="three_seeds"

# FINETUNING_DATASETS_LIST="manual_in_python_script"
# FINETUNING_DATASETS_LIST="multiwoz21_and_reddit_full"
# FINETUNING_DATASETS_LIST="multiwoz21_full"
FINETUNING_DATASETS_LIST="reddit_full"

# Uncomment the following to skip the compute_and_store_embeddings step:
#
# SKIP_COMPUTE_AND_STORE_EMBEDDINGS="--skip_compute_and_store_embeddings"

EMBEDDINGS_DATA_PREP_SAMPLING_MODE="random"
EMBEDDINGS_DATA_PREP_SAMPLING_SEED_LIST_OPTION="five_seeds"

LOCAL_ESTIMATES_FILTERING_NUM_SAMPLES_LIST="default"
LOCAL_ESTIMATES_POINTWISE_ABSOLUTE_N_NEIGHBORS_LIST="single_choice_128"


# ================================================================== #

# ---------------------------------------------------------- #
# Experiment setup.
# Note that the individual experiment setup can override these values.
USE_COMMON_EXPERIMENT_SETUP="true"

# EXPERIMENT_SELECTOR="multiwoz21_different_data_subsampling_number_of_samples"
EXPERIMENT_SELECTOR="reddit_different_data_subsampling_number_of_samples"

# EXPERIMENT_SELECTOR="multiwoz21_different_checkpoints"
# EXPERIMENT_SELECTOR="reddit_different_checkpoints"

# EXPERIMENT_SELECTOR="multiwoz21_and_reddit_data_subsampling_take_first_different_checkpoints"

# ---------------------------------------------------------- #

# EXPERIMENT_STAGE="compute_embeddings_plus_single_pipeline_run"
EXPERIMENT_STAGE="skip_compute_embeddings_and_multiple_pipeline_runs"

# ---------------------------------------------------------- #

if [ "${USE_COMMON_EXPERIMENT_SETUP}" = "true" ]; then
  DATA_SUBSAMPLING_SAMPLING_MODE="random"
  DATA_SUBSAMPLING_SAMPLING_SEED_LIST_OPTION="three_seeds"

  EMBEDDINGS_DATA_PREP_SAMPLING_MODE="random"
  EMBEDDINGS_DATA_PREP_NUM_SAMPLES_LIST_OPTION="single_choice_150000"

  LOCAL_ESTIMATES_POINTWISE_ABSOLUTE_N_NEIGHBORS_LIST="single_choice_128"
fi

echo ">>> Experiment selected: ${EXPERIMENT_SELECTOR}"

# ================================================================== #
#
# ++++ Experiment > different subsampling number of samples for multiwoz21 dataset
if [ "${EXPERIMENT_SELECTOR}" = "multiwoz21_different_data_subsampling_number_of_samples" ]; then
  DATA_LIST="multiwoz21_only"
  # DATA_SUBSAMPLING_NUMBER_OF_SAMPLES_LIST_OPTION="range_start_2000_stop_18000_step_2000"
  # DATA_SUBSAMPLING_NUMBER_OF_SAMPLES_LIST_OPTION="range_start_12000_stop_18000_step_2000"
  DATA_SUBSAMPLING_NUMBER_OF_SAMPLES_LIST_OPTION="fixed_16000"
  
  LOCAL_ESTIMATES_FILTERING_NUM_SAMPLES_LIST="single_choice_60000"

  # Note: We explicitly increase the memory size here,
  # since for the embeddings data prep step on 12_000 and more data subsamlping samples,
  # the embeddings data prep step requires more memory.
  MEMORY="64"
fi

# ++++ Experiment > different subsampling number of samples for reddit dataset
if [ "${EXPERIMENT_SELECTOR}" = "reddit_different_data_subsampling_number_of_samples" ]; then
  DATA_LIST="reddit_only"
  # DATA_SUBSAMPLING_NUMBER_OF_SAMPLES_LIST_OPTION="range_start_2000_stop_24000_step_2000"
  # DATA_SUBSAMPLING_NUMBER_OF_SAMPLES_LIST_OPTION="range_start_12000_stop_24000_step_2000"
  DATA_SUBSAMPLING_NUMBER_OF_SAMPLES_LIST_OPTION="fixed_22000"

  LOCAL_ESTIMATES_FILTERING_NUM_SAMPLES_LIST="single_choice_60000"

  # Note: We explicitly increase the memory size here,
  # since for the embeddings data prep step on 12_000 and more data subsamlping samples,
  # the embeddings data prep step requires more memory.
  MEMORY="80"
fi

# ================================================================== #
#
# ++++ Experiment > different checkpoints for multiwoz21 dataset
if [ "${EXPERIMENT_SELECTOR}" = "multiwoz21_different_checkpoints" ]; then
  DATA_LIST="multiwoz21_only"
  DATA_SUBSAMPLING_NUMBER_OF_SAMPLES_LIST_OPTION="fixed_10000"

  CHECKPOINT_NO_LIST="selected"

  LOCAL_ESTIMATES_FILTERING_NUM_SAMPLES_LIST="single_choice_60000"
fi

# ++++ Experiment > different checkpoints for reddit dataset
if [ "${EXPERIMENT_SELECTOR}" = "reddit_different_checkpoints" ]; then
  DATA_LIST="reddit_only"
  DATA_SUBSAMPLING_NUMBER_OF_SAMPLES_LIST_OPTION="fixed_10000"

  CHECKPOINT_NO_LIST="selected"

  LOCAL_ESTIMATES_FILTERING_NUM_SAMPLES_LIST="single_choice_60000"
fi

# ================================================================== #
#
# ++++ Experiment > different checkpoints for multiwoz21 dataset

if [ "${EXPERIMENT_SELECTOR}" = "multiwoz21_and_reddit_data_subsampling_take_first_different_checkpoints" ]; then
  # Select 10_000 from the dataset with the take_first sampling mode.
  # Note that in this case, we need to set the data_subsampling_sampling_seed_list_option to none.
  DATA_LIST="multiwoz21_and_reddit"

  DATA_SUBSAMPLING_SAMPLING_MODE="take_first"
  DATA_SUBSAMPLING_SAMPLING_SEED_LIST_OPTION="none"
  DATA_SUBSAMPLING_NUMBER_OF_SAMPLES_LIST_OPTION="fixed_10000"

  CHECKPOINT_NO_LIST="selected"

  EMBEDDINGS_DATA_PREP_SAMPLING_MODE="random"
  EMBEDDINGS_DATA_PREP_NUM_SAMPLES_LIST_OPTION="single_choice_150000"

  LOCAL_ESTIMATES_POINTWISE_ABSOLUTE_N_NEIGHBORS_LIST="single_choice_128"
fi

# ================================================================== #
#
echo ">>> Experiment stage selected: ${EXPERIMENT_STAGE}"

if [ "${EXPERIMENT_STAGE}" = "compute_embeddings_plus_single_pipeline_run" ]; then
  NCPUS="4"
  NGPUS="1"
  QUEUE="CUDA"
  
  # TEMPLATE="TESLAT4"
  # TEMPLATE="GTX1080"
  TEMPLATE="RTX6000"

  EMBEDDINGS_DATA_PREP_SAMPLING_SEED_LIST_OPTION="default"
  
  SKIP_COMPUTE_AND_STORE_EMBEDDINGS="" # do the embeddings computation
elif [ "${EXPERIMENT_STAGE}" = "skip_compute_embeddings_and_multiple_pipeline_runs" ]; then
  NCPUS="6"
  NGPUS="0"
  QUEUE="DEFAULT"
  TEMPLATE="CPU"

  EMBEDDINGS_DATA_PREP_SAMPLING_SEED_LIST_OPTION="five_seeds"

  SKIP_COMPUTE_AND_STORE_EMBEDDINGS="--skip_compute_and_store_embeddings" # skip the embeddings computation
fi  

# ---------------------------------------------------------- #

# ================================================================== #
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
      --template=$TEMPLATE \
      --queue=$QUEUE \
      --memory=$MEMORY \
      --ncpus=$NCPUS \
      --ngpus=$NGPUS \
      --data_list=$DATA_LIST \
      --data_subsampling_sampling_mode=$DATA_SUBSAMPLING_SAMPLING_MODE \
      --data_subsampling_number_of_samples_list_option=$DATA_SUBSAMPLING_NUMBER_OF_SAMPLES_LIST_OPTION \
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
      $RUN_ONLY_FIRST_CONFIG_OPTION_FLAG \
      $DRY_RUN_FLAG
  echo ">>> Submitting pipeline jobs DONE"
fi
# ================================================================== #
       
# ================================================================== #
if [ "$DO_LOCAL_ESTIMATES_COMPUTATION" = "true" ]; then
  echo ">>> Overwriting TEMPLATE; QUEUE; NGPUS for local estimates computation ..."
  TEMPLATE="CPU"
  QUEUE="DEFAULT"
  NGPUS="0"
  echo ">>> Overwritten values: TEMPLATE=${TEMPLATE}; QUEUE=${QUEUE}; NGPUS=${NGPUS}."

  echo ">>> Submitting local estimates computation jobs ..."
  # This is a CPU task, so we do not ask for a GPU.
  poetry run submit_jobs \
      --task="local_estimates_computation" \
      --template=$TEMPLATE \
      --queue=$QUEUE \
      --ncpus=$NCPUS \
      --ngpus=$NGPUS \
      --walltime="08:00:00" \
      --data_list=$DATA_LIST \
      --data_subsampling_sampling_mode=$DATA_SUBSAMPLING_SAMPLING_MODE \
      --data_subsampling_number_of_samples_list_option=$DATA_SUBSAMPLING_NUMBER_OF_SAMPLES_LIST_OPTION \
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
      $RUN_ONLY_FIRST_CONFIG_OPTION_FLAG \
      $DRY_RUN_FLAG
  echo ">>> Submitting local estimates computation jobs DONE"
fi
# ================================================================== #

# ================================================================== #
if [ "$DO_PERPLEXITY" = "true" ]; then
  echo ">>> Overwriting TEMPLATE and QUEUE for perplexity computation ..."
  TEMPLATE="CUDA"
  QUEUE="RTX6000"
  echo ">>> Overwritten values: QUEUE=${QUEUE}, TEMPLATE=${TEMPLATE}."

  echo ">>> Submitting perplexity jobs ..."
  # Note: Do not use the CREATE_POS_TAGS_FLAG for the perplexity task.
  poetry run submit_jobs \
      --task="perplexity" \
      --template=$TEMPLATE \
      --queue=$QUEUE \
      --ncpus=$NCPUS \
      --ngpus=$NGPUS \
      --data_list=$DATA_LIST \
      --data_subsampling_sampling_mode=$DATA_SUBSAMPLING_SAMPLING_MODE \
      --data_subsampling_number_of_samples_list_option=$DATA_SUBSAMPLING_NUMBER_OF_SAMPLES_LIST_OPTION \
      --data_subsampling_sampling_seed_list_option=$DATA_SUBSAMPLING_SAMPLING_SEED_LIST_OPTION \
      --language_model_list=$LANGUAGE_MODEL_LIST \
      --checkpoint_no_list=$CHECKPOINT_NO_LIST \
      --language_model_seed_list=$LANGUAGE_MODEL_SEED_LIST \
      $ADD_PREFIX_SPACE_FLAG \
      --finetuning_regime=$FINETUNING_REGIME \
      --submission_mode=$SUBMISSION_MODE \
      $RUN_ONLY_FIRST_CONFIG_OPTION_FLAG \
      $DRY_RUN_FLAG
  echo ">>> Submitting perplexity jobs DONE"
fi
# ================================================================== #

# ================================================================== #
# Notes on memory size:
#
# ++ accelerator_model=rtx6000:
#   + `--common_batch_size="32"` appears to work for fine-tuning "roberta-base" model on rtx6000 with 24GB of VRAM.

if [ "$DO_FINETUNING" = "true" ]; then
  echo ">>> Overwriting TEMPLATE and QUEUE for perplexity computation ..."
  TEMPLATE="CUDA"
  QUEUE="RTX6000"
  echo ">>> Overwritten values: QUEUE=${QUEUE}, TEMPLATE=${TEMPLATE}."

  echo ">>> Submitting finetuning jobs ..."
  poetry run submit_jobs \
    --task="finetuning" \
    --template=$TEMPLATE \
    --queue=$QUEUE \
    --memory=$MEMORY \
    --ncpus=$NCPUS \
    --ngpus=$NGPUS \
    --walltime="48:00:00" \
    --finetuning_datasets_list=$FINETUNING_DATASETS_LIST \
    --finetuning_seed_list="one_seed" \
    --finetuning_regime="many_epochs_with_overfitting_risk" \
    --common_batch_size="32" \
    --submission_mode=$SUBMISSION_MODE \
    --wandb_project="Topo_LLM_finetuning_from_submission_script_DEBUG_large_batch_size" \
    $RUN_ONLY_FIRST_CONFIG_OPTION_FLAG \
    $DRY_RUN_FLAG
  echo ">>> Submitting finetuning jobs ..."
fi
# ================================================================== #

# ================================================================== #
# Exit submission script
echo ">>> Submission script finished."
echo ">>> Exiting ..."
exit $?
# ================================================================== #
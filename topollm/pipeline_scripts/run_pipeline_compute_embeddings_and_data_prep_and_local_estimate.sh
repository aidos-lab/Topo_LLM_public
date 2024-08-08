#!/bin/bash

echo "TOPO_LLM_REPOSITORY_BASE_PATH=${TOPO_LLM_REPOSITORY_BASE_PATH}"

PYTHON_SCRIPT_NAME="run_pipeline_compute_embeddings_and_data_prep_and_local_estimate.py"
RELATIVE_PYTHON_SCRIPT_PATH="topollm/pipeline_scripts/${PYTHON_SCRIPT_NAME}"
ABSOLUTE_PYTHON_SCRIPT_PATH="${TOPO_LLM_REPOSITORY_BASE_PATH}/${RELATIVE_PYTHON_SCRIPT_PATH}"


# # # # # # # # # # # # # # # # # # # # # # # # # # #
# START: Python script - Command line arguments
#
# We choose parameters here which can be used for debugging the script in a reasonable time.

# DATA_LIST="multiwoz21_validation,iclr_2024_submissions,wikitext"
DATA_LIST="multiwoz21_validation"

LANGUAGE_MODEL_LIST="roberta-base"
# LANGUAGE_MODEL_LIST="roberta-base_finetuned-on-iclr_ftm-standard"

CHECKPOINT_NO="400"
# CHECKPOINT_NO="400,800,1200"

LAYER_INDICES_LIST="[-1]"
# LAYER_INDICES_LIST="[-1],[-2]"

# DATA_NUMBER_OF_SAMPLES="128"
DATA_NUMBER_OF_SAMPLES="3000"

# EMBEDDINGS_DATA_PREP_NUM_SAMPLES="1000"
EMBEDDINGS_DATA_PREP_NUM_SAMPLES="30000"
# EMBEDDINGS_DATA_PREP_NUM_SAMPLES="10000,20000"

ADDITIONAL_OVERRIDES=""

# END: Python script - Command line arguments
# # # # # # # # # # # # # # # # # # # # # # # # # # #

# ==================================================== #

echo "Calling python script ABSOLUTE_PYTHON_SCRIPT_PATH=${ABSOLUTE_PYTHON_SCRIPT_PATH} ..."

poetry run python3 $ABSOLUTE_PYTHON_SCRIPT_PATH \
    --multirun \
    data=$DATA_LIST \
    +data.dataset_type=huggingface_dataset_named_entity \
    language_model=$LANGUAGE_MODEL_LIST \
    +language_model.checkpoint_no=$CHECKPOINT_NO \
    embeddings.embedding_extraction.layer_indices=$LAYER_INDICES_LIST \
    data.number_of_samples=$DATA_NUMBER_OF_SAMPLES \
    embeddings_data_prep.num_samples=$EMBEDDINGS_DATA_PREP_NUM_SAMPLES \
    $ADDITIONAL_OVERRIDES


echo "Finished python script."

# ==================================================== #

# Exit with return code from the last command.
exit $?
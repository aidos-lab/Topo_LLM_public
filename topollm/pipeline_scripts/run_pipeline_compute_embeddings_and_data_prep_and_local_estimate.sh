#!/bin/bash

echo "TOPO_LLM_REPOSITORY_BASE_PATH=${TOPO_LLM_REPOSITORY_BASE_PATH}"

PYTHON_SCRIPT_NAME="run_pipeline_compute_embeddings_and_data_prep_and_local_estimate.py"
RELATIVE_PYTHON_SCRIPT_PATH="topollm/pipeline_scripts/${PYTHON_SCRIPT_NAME}"
ABSOLUTE_PYTHON_SCRIPT_PATH="${TOPO_LLM_REPOSITORY_BASE_PATH}/${RELATIVE_PYTHON_SCRIPT_PATH}"

# >> Local run
# HYDRA_LAUNCHER_ARGS_LIST=(
#     "hydra/launcher=basic"
#     "preferred_torch_backend=cpu" # Locally on a MacBook with 16GB of memory, loading Phi-3.5 on MPS is not possible
# )
# >> HPC run
HYDRA_LAUNCHER_ARGS_LIST=(
    "hydra/launcher=hpc_submission"
    "hydra.launcher.queue=CUDA"
    "hydra.launcher.template=RTX6000" # Note: 12 GB of GPU memory appears to not be enough for the GPT-2 pipeline, i.e., do not select the "hydra.launcher.template=GTX1080"
    "hydra.launcher.memory=50" # <-- The embeddings data prep step failed with 32GB of memory for the GPT2 medium model. 
    "hydra.launcher.ncpus=2" # <-- Make sure not to use more than 2 CPUs per GPU on the GTX1080TI and RTX6000 nodes.
    "hydra.launcher.ngpus=1"
    "hydra.launcher.walltime=04:00:00" # <-- The pipeline run for regular embeddings should only take a few minutes.
)

# # # # # # # # # # # # # # # # # # # # # # # # # # #
# START: Python script - Command line arguments
#
# We choose parameters here which can be used for debugging the script in a reasonable time.

DATASET_TYPE_LINE=""
# DATASET_TYPE_LINE="+data.dataset_type=huggingface_dataset_named_entity"

# DATA_LIST="multiwoz21_validation,iclr_2024_submissions,wikitext"
DATA_LIST="multiwoz21_validation"

# LANGUAGE_MODEL_LIST="roberta-base"
# LANGUAGE_MODEL_LIST="roberta-base_finetuned-on-iclr_ftm-standard"

# Note:
# Currently there is a problem with the "ModernBERT-base" model:
# > RuntimeError: Failed to import transformers.models.modernbert.modeling_modernbert because of the following error (look up to see its traceback):
# > Dynamo is not supported on Python 3.12+
#
# LANGUAGE_MODEL_LIST="ModernBERT-base"

# LANGUAGE_MODEL_LIST="Phi-3.5-mini-instruct"
LANGUAGE_MODEL_LIST="Phi-4-mini-instruct"

CHECKPOINT_NO="-1"
# CHECKPOINT_NO="400"
# CHECKPOINT_NO="400,800,1200"

# Notes: 
# - For larger models, we need to reduce the batch size to avoid OOM errors on the GPU.
EMBEDDINGS_BATCH_SIZE="2"
# EMBEDDINGS_BATCH_SIZE="32" # We have used a batch size of 32 for RoBERTa-base and GPT-2-medium models.

# Notes:
# - The embedding dimension of the "Phi-3.5-mini-instruct" model is 3072.
#   To save the array data of this model, we need to reduce the chunk size in the Zarr storage format to avoid errors like:
#   "Codec does not support buffers of > 2147483647 bytes"
#   https://github.com/zarr-developers/zarr-python/issues/487
STORAGE_CHUNK_SIZE="32"

LAYER_INDICES_LIST="[-1]"
# LAYER_INDICES_LIST="[-1],[-2]"

# DATA_NUMBER_OF_SAMPLES="128"
DATA_NUMBER_OF_SAMPLES="512"
# DATA_NUMBER_OF_SAMPLES="3000"

# EMBEDDINGS_DATA_PREP_NUM_SAMPLES="1000"
EMBEDDINGS_DATA_PREP_NUM_SAMPLES="30000"
# EMBEDDINGS_DATA_PREP_NUM_SAMPLES="10000,20000"

ADDITIONAL_OVERRIDES=""

# END: Python script - Command line arguments
# # # # # # # # # # # # # # # # # # # # # # # # # # #

# ==================================================== #

echo ">>> Calling python script ABSOLUTE_PYTHON_SCRIPT_PATH=${ABSOLUTE_PYTHON_SCRIPT_PATH} ..."

uv run python3 $ABSOLUTE_PYTHON_SCRIPT_PATH \
    --multirun \
    "hydra/sweeper=basic" \
    "${HYDRA_LAUNCHER_ARGS_LIST[@]}" \
    data=$DATA_LIST \
    $DATASET_TYPE_LINE \
    language_model=$LANGUAGE_MODEL_LIST \
    +language_model.checkpoint_no=$CHECKPOINT_NO \
    embeddings.batch_size=$EMBEDDINGS_BATCH_SIZE \
    embeddings.embedding_extraction.layer_indices=$LAYER_INDICES_LIST \
    "data.data_subsampling.number_of_samples=$DATA_NUMBER_OF_SAMPLES" \
    embeddings_data_prep.sampling.num_samples=$EMBEDDINGS_DATA_PREP_NUM_SAMPLES \
    "embeddings_data_prep.sampling.sampling_mode=random" \
    "local_estimates=twonn" \
    "local_estimates.filtering.num_samples=500" \
    "local_estimates.compute_global_estimates=True" \
    "local_estimates.pointwise.n_neighbors_mode=absolute_size" \
    "local_estimates.pointwise.absolute_n_neighbors=128" \
    "storage.chunk_size=$STORAGE_CHUNK_SIZE" \
    $ADDITIONAL_OVERRIDES


echo ">>> Finished python script."

# ==================================================== #

# Exit with return code from the last command.
exit $?
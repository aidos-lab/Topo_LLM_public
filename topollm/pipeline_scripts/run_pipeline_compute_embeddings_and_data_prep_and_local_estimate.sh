#!/bin/bash

# # # # # # # # # # # # # # # # # # # # # # # # #
# Default values

# Initialize flags / required parameters
dry_run=false
launcher=""  # <-- REQUIRED. Must be provided via --launcher <basic|basic_cpu|hpc_submission>

# # # # # # # # # # # # # # # # # # # # # # # # #
# Function to print usage
usage() {
    echo "💡 Usage: $0 --launcher <basic|basic_cpu|hpc_submission> [--dry-run] [additional hydra overrides ...]"
    echo "    --launcher <value> : REQUIRED. Select launcher type ('basic', 'basic_cpu', or 'hpc_submission')."
    echo "    --dry-run          : Only print the constructed command, do not execute it."
    echo "    [additional args]  : Any further arguments are passed through as Hydra overrides."
    exit 1
}

POSITIONAL_ARGS=()
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dry-run)
            dry_run=true
            shift
            ;;
        --launcher)
            if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                launcher="$2"
                shift 2
            else
                echo "❌ Error: --launcher requires a value ('basic' | 'basic_cpu' | 'hpc_submission')"
                usage
            fi
            ;;
        --help|-h)
            usage
            ;;
        --*)
            echo "❌ Error: Unknown option: $1"
            usage
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL_ARGS[@]}"

# Validate required parameters
if [[ -z "$launcher" ]]; then
    echo "❌ Error: --launcher argument is required."
    usage
fi

# # # # # # # # # # # # # # # # # # # # # # # # #
# Check if environment variables are set

if [[ -z "${TOPO_LLM_REPOSITORY_BASE_PATH}" ]]; then
    echo "❌ Error: TOPO_LLM_REPOSITORY_BASE_PATH is not set."
    exit 1
fi

# Load environment variables from the .env file
source "${TOPO_LLM_REPOSITORY_BASE_PATH}/.env"

# # # # # # # # # # # # # # # # # # # # # # # # #
# Log variables

VARIABLES_TO_LOG_LIST=(
    "TOPO_LLM_REPOSITORY_BASE_PATH" # example: /gpfs/project/$USER/git-source/Topo_LLM
)

for VARIABLE_NAME in "${VARIABLES_TO_LOG_LIST[@]}"; do
    echo "💡 ${VARIABLE_NAME}=${!VARIABLE_NAME}"
done

PYTHON_SCRIPT_NAME="run_pipeline_compute_embeddings_and_data_prep_and_local_estimate.py"
RELATIVE_PYTHON_SCRIPT_PATH="topollm/pipeline_scripts/${PYTHON_SCRIPT_NAME}"
ABSOLUTE_PYTHON_SCRIPT_PATH="${TOPO_LLM_REPOSITORY_BASE_PATH}/${RELATIVE_PYTHON_SCRIPT_PATH}"

echo "===================================================="
echo ">> Parsed arguments:"
echo "===================================================="
echo "  dry_run: ${dry_run}"
echo "  launcher: ${launcher}"
echo "  positional args (passed through): ${POSITIONAL_ARGS[*]}"
echo "===================================================="


# ==================================================== #
# >>> START Select parameters

BASE_ARGS=(
    "--multirun"
    "hydra/sweeper=basic"
)

# Launcher selection (required --launcher argument)
#
# Notes: 
# - Locally on a MacBook Pro M1 with 16GB of memory, 
#   loading Phi-3.5 on MPS is not possible.
#
case "$launcher" in
    basic)
        # >> Local run
        LAUNCHER_ARGS=(
            "hydra/launcher=basic"
            "preferred_torch_backend=auto"
        )
        ;;
    basic_cpu)
        # >> Local run
        # Same as 'basic' except for explicit CPU torch backend selection
        LAUNCHER_ARGS=(
            "hydra/launcher=basic"
            "preferred_torch_backend=cpu"
        )
        ;;
    hpc_submission)
        # >> HPC run
        LAUNCHER_ARGS=(
            "hydra/launcher=hpc_submission"
            "hydra.launcher.queue=CUDA"
            # >>> GPU selection
            # > Notes: 
            # > - 12 GB of GPU memory appears to not be enough for the GPT-2 pipeline, i.e., do not select "hydra.launcher.template=GTX1080" for GPT-2.
            # > - 24 GB of GPU memory is enough for the embedding computation for the 1B and 3B variants of the Llama-3 models,
            # >   but NOT enough for the 8B variant. For the 8B variant, do NOT select "hydra.launcher.template=RTX6000". 
            "hydra.launcher.template=RTX6000"
            # "hydra.launcher.template=RTX8000"
            # >>> Other resources
            "hydra.launcher.memory=50"   # <-- The embeddings data prep step failed with 32GB of memory for the GPT2 medium model.
            "hydra.launcher.ncpus=2"     # <-- Make sure not to use more than 2 CPUs per GPU on the GTX1080TI and RTX6000 nodes.
            "hydra.launcher.ngpus=1"
            # The pipeline run for regular embeddings can take longer than an hour for large models.
            "hydra.launcher.walltime=04:00:00" 
        )
        ;;
    *)
        echo "❌ Error: Unknown launcher value '$launcher' (expected one of: basic, basic_cpu, hpc_submission)."
        usage
        ;;
esac

# # # # # # # # # # # # # # # # # # # # # # # # # # #
# START: Python script - Command line arguments
#
# We choose parameters here which can be used for debugging the script in a reasonable time.

DATASET_TYPE_LINE=""
# DATASET_TYPE_LINE="+data.dataset_type=huggingface_dataset_named_entity"

DATA_ARGS=(
    # "data=multiwoz21_validation"
    # "data=multiwoz21_validation,iclr_2024_submissions,wikitext"
    # TODO: Create concatenated 'source' + 'target' column for LUSTER data
    # TODO: Use concatenated 'source' + 'target' for LUSTER data
    # TODO: Implement the token mask for the local estimates computation
    "data=luster"
    "data.column_name=source"
    # "data.column_name=target"
    $DATASET_TYPE_LINE
)
 
# ===== RoBERTa and BERT models ===== #

# LANGUAGE_MODEL_LIST="roberta-base"
# LANGUAGE_MODEL_LIST="roberta-base_finetuned-on-iclr_ftm-standard"

# Note:
# Currently there is a problem with the "ModernBERT-base" model:
# > RuntimeError: Failed to import transformers.models.modernbert.modeling_modernbert because of the following error (look up to see its traceback):
# > Dynamo is not supported on Python 3.12+
#
# LANGUAGE_MODEL_LIST="ModernBERT-base"

# ===== GPT-2 models ===== #

# LANGUAGE_MODEL_LIST="gpt2" # <-- Smallest GPT-2 model with 124M parameters (Safetensors: 137M parameters).

# ===== Phi-3.5 and Phi-4 models ===== #
# https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3

# LANGUAGE_MODEL_LIST="Phi-3.5-mini-instruct" # <-- "microsoft/Phi-3.5-mini-instruct" model with 3.8B parameters (Safetensors: 3.82B parameters).
# LANGUAGE_MODEL_LIST="Phi-4-mini-instruct"

# ===== LUSTER models ===== #
# https://arxiv.org/abs/2507.01594
#
# Notes:
# - These models are fine-tuned versions of the "microsoft/Phi-3.5-mini-instruct" model,
#   but they are usually supposed to be used with a certain contrained decoding strategy.
#   Nevertheless, for testing. we plug them into our inference setup here.

# > Base model:
LANGUAGE_MODEL_LIST="Phi-3.5-mini-instruct"

# LANGUAGE_MODEL_LIST="luster-base"
# LANGUAGE_MODEL_LIST="luster-base-emotion"
# LANGUAGE_MODEL_LIST="luster-chitchat"
# LANGUAGE_MODEL_LIST="luster-full"
# LANGUAGE_MODEL_LIST="luster-rl-sent"
# LANGUAGE_MODEL_LIST="luster-rl-succ"

# ===== Gemma models ===== #
# https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d

# LANGUAGE_MODEL_LIST="gemma-3-1b-pt" # <-- (Safetensors: 1,000M parameters)
# LANGUAGE_MODEL_LIST="gemma-3-1b-it" # <-- (Safetensors: 1,000M parameters)

# ===== LLama models ===== #

# Notes:
# - There is no 8B variant of the Llama-3.2 models, this only exists for the Llama-3.1 models.

# LANGUAGE_MODEL_LIST="Llama-3.2-1B" # <-- (Safetensors: 1.24B parameters)
# LANGUAGE_MODEL_LIST="Llama-3.2-3B" # <-- (Safetensors: 3.21B parameters)

# LANGUAGE_MODEL_LIST="Llama-3.1-8B" # <-- (Safetensors: 8.03B parameters)

# Multiple Llama-3 models:
# LANGUAGE_MODEL_LIST="Llama-3.2-1B,Llama-3.2-3B,Llama-3.1-8B"

# # # # # # # # # # # # # # # # # # # #

CHECKPOINT_NO="-1"
# CHECKPOINT_NO="400"
# CHECKPOINT_NO="400,800,1200"
# CHECKPOINT_NO="400,800,1200,1600,2000,2400,2800"

LANGUAGE_MODEL_ARGS=(
    "language_model=$LANGUAGE_MODEL_LIST"
    "++language_model.checkpoint_no=$CHECKPOINT_NO"
)

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

EMBEDDINGS_ARGS=(
    "embeddings.batch_size=$EMBEDDINGS_BATCH_SIZE"
    "embeddings.embedding_extraction.layer_indices=$LAYER_INDICES_LIST"
)

# DATA_NUMBER_OF_SAMPLES="16"
# DATA_NUMBER_OF_SAMPLES="128"
DATA_NUMBER_OF_SAMPLES="512"
# DATA_NUMBER_OF_SAMPLES="3000"

# EMBEDDINGS_DATA_PREP_NUM_SAMPLES="1000"
EMBEDDINGS_DATA_PREP_NUM_SAMPLES="30000"
# EMBEDDINGS_DATA_PREP_NUM_SAMPLES="10000,20000"

SAMPLING_ARGS=(
    "data.data_subsampling.number_of_samples=$DATA_NUMBER_OF_SAMPLES"
    "embeddings_data_prep.sampling.num_samples=$EMBEDDINGS_DATA_PREP_NUM_SAMPLES"
    "embeddings_data_prep.sampling.sampling_mode=random"
)

LOCAL_ESTIMATES_ARGS=(
    "local_estimates=twonn"
    "local_estimates.filtering.num_samples=500"
    "local_estimates.compute_global_estimates=True"
    "local_estimates.pointwise.n_neighbors_mode=absolute_size"
    "local_estimates.pointwise.absolute_n_neighbors=128"
)

STORAGE_ARGS=(
    "storage.chunk_size=$STORAGE_CHUNK_SIZE"
)

FEATURE_FLAGS_ARGS=(
    # NOTE: Only skip the embedding computation if the embeddings are already available.
    "feature_flags.compute_and_store_embeddings.skip_compute_and_store_embeddings_in_pipeline=False"
    "feature_flags.analysis.create_plots_in_local_estimates_worker=True"
)

# END: Python script - Command line arguments
# # # # # # # # # # # # # # # # # # # # # # # # # # #

ADDITIONAL_OVERRIDES=""

# >>> END: Select parameters
# ==================================================== #

COMMON_ARGS=(
    $ADDITIONAL_OVERRIDES
)

ARGS=(
    "${BASE_ARGS[@]}"
    "${LAUNCHER_ARGS[@]}"
    "${DATA_ARGS[@]}"
    "${LANGUAGE_MODEL_ARGS[@]}"
    "${EMBEDDINGS_ARGS[@]}"
    "${SAMPLING_ARGS[@]}"
    "${LOCAL_ESTIMATES_ARGS[@]}"
    "${STORAGE_ARGS[@]}"
    "${FEATURE_FLAGS_ARGS[@]}"
    "${COMMON_ARGS[@]}"
    "$@"
)

# Note:
# Do NOT set the CUDA_VISIBLE_DEVICES environment variable on the HPC cluster.
# > "++hydra.job.env_set.CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# CUDA_VISIBLE_DEVICES=0
echo ">>> CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# ==================================================== #

# Print the argument list
echo "===================================================="
echo ">> Argument list:"
echo "===================================================="
for arg in "${ARGS[@]}"; do
    echo "  $arg"
done
echo "===================================================="

echo ">>> Calling script from RELATIVE_PYTHON_SCRIPT_PATH=${RELATIVE_PYTHON_SCRIPT_PATH} ..."

if [ "$dry_run" = true ]; then
    echo "💡 [DRY RUN] Would run: uv run python3 $RELATIVE_PYTHON_SCRIPT_PATH ${ARGS[*]}"
else
    uv run python3 $RELATIVE_PYTHON_SCRIPT_PATH "${ARGS[@]}"
fi

echo ">>> Calling script from RELATIVE_PYTHON_SCRIPT_PATH=${RELATIVE_PYTHON_SCRIPT_PATH} DONE"

# ==================================================== #

# Exit with last command's exit code
echo ">>> Exiting with code $?."
exit $?
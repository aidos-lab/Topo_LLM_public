#!/bin/bash

# # # # # # # # # # # # # # # # # # # # # # # # #
# Default values

# Initialize dry_run flag to false
dry_run=false

# # # # # # # # # # # # # # # # # # # # # # # # #
# Function to print usage
usage() {
    echo "💡 Usage: $0 [--dry-run]"
    exit 1
}

# Parse command line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dry-run)
            dry_run=true
            shift # Remove the --dry_run argument
            ;;
        *)
            echo "❌ Error: Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

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

PYTHON_SCRIPT_NAME="run_inference_pipeline.py"
RELATIVE_PYTHON_SCRIPT_PATH="topollm/model_inference/${PYTHON_SCRIPT_NAME}"
ABSOLUTE_PYTHON_SCRIPT_PATH="${TOPO_LLM_REPOSITORY_BASE_PATH}/${RELATIVE_PYTHON_SCRIPT_PATH}"


# ==================================================== #
# >>> START Select parameters
#
# Notes on memory usage:
# - "Phi-3.5-mini-instruct":
#   (3.82B parameters, "hidden_size": 3072)
#   24 GB of VRAM is enough for the model and fine-tuned variants.
#   Inference is possible on a single RTX6000-24GB GPU.
# - "Llama-3.1-8B":
#   (8.03B parameters, "hidden_size": 4096)
#   24 GB of VRAM is NOT enough, and leads to CUDA out of memory errors.
#   48 GB of VRAM is enough for the model and fine-tuned variants.
#   Inference is possible on a single RTX8000-48GB GPU.
#   Do NOT use an RTX6000-24GB GPU for this model, as it will lead to CUDA out of memory errors.
#
# Example commands for starting interactive sessions on the HPC cluster:
#
# - RTX6000-24GB:
# > qsub -I -N DevSession -A DialSys -q CUDA -l select=1:ncpus=2:mem=32gb:ngpus=1:accelerator_model=rtx6000 -l walltime=8:00:00
#
# - RTX8000-48GB:
# > qsub -I -N DevSession -A DialSys -q CUDA -l select=1:ncpus=2:mem=32gb:ngpus=1:accelerator_model=rtx8000 -l walltime=8:00:00

BASE_ARGS=(
    "--multirun"
    "hydra/sweeper=basic"
)

HYDRA_LAUNCHER_ARGS=(
    "hydra/launcher=hpc_submission"
    "hydra.launcher.queue=CUDA"
    # >>> GPU selection
    # "hydra.launcher.template=RTX6000"
    "hydra.launcher.template=RTX8000"
    # >>> Other resources
    "hydra.launcher.memory=64"
    "hydra.launcher.ncpus=2"
    "hydra.launcher.ngpus=1"
    # The model inference is fast, so we can set a shorter walltime.
    "hydra.launcher.walltime=00:30:00"
)

# Notes:
# - Model parameter counts are taking from the Hugging Face model card from the 'Safetensors' section.
#   Thus, the counts might differ from the model card's 'Model description' section.


# ===== RoBERTa and BERT models ===== #

# LANGUAGE_MODEL_LIST="roberta-base"
# LANGUAGE_MODEL_LIST="bert-base-uncased,roberta-base,gpt2-large"
# LANGUAGE_MODEL_LIST="roberta-base,roberta-base_finetuned-on-multiwoz21_ftm-lora"
# LANGUAGE_MODEL_LIST="bert-base-uncased,bert-base-uncased_finetuned-on-xsum_ftm-standard,bert-base-uncased_finetuned-on-multiwoz21_ftm-standard,bert-base-uncased_finetuned-on-sgd_ftm-lora"
# LANGUAGE_MODEL_LIST="roberta-base_finetuned-on-one-year-of-tsla-on-reddit_ftm-standard_freeze-first-6-layers_overfitted"
# LANGUAGE_MODEL_LIST="roberta-base_finetuned-on-multiwoz21_ftm-standard_full-dataset"

# ===== GPT-2 models ===== #

# LANGUAGE_MODEL_LIST="gpt2" # <-- Smallest GPT-2 model with 124M parameters (Safetensors: 137M parameters).

# LANGUAGE_MODEL_LIST="gpt2-medium" # <-- Medium GPT-2 model with 355M parameters (Safetensors: 380M parameters).

# LANGUAGE_MODEL_LIST="gpt2-medium-causal_lm-defaults_multiwoz21-rm-empty-True-do_nothing-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5"
# LANGUAGE_MODEL_LIST="gpt2-medium-causal_lm-defaults_one-year-of-tsla-on-reddit-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5"

# LANGUAGE_MODEL_LIST="gpt2-large" # <-- Large GPT-2 model with 774M parameters (Safetensors: 812M parameters).

# ===== Phi-3.5 and Phi-4 models ===== #
# https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3

# LANGUAGE_MODEL_LIST="Phi-3.5-mini-instruct" # <-- "microsoft/Phi-3.5-mini-instruct" model with 3.8B parameters (Safetensors: 3.82B parameters).

# LANGUAGE_MODEL_LIST="Phi-3.5-mini-instruct-causal_lm-defaults_multiwoz21-rm-empty-True-do_nothing-ner_tags_train-10000-random-778_aps-False-mx-512_lora-16-32-o_proj_qkv_proj-0.01-True_5e-05-linear-0.01-5"
# LANGUAGE_MODEL_LIST="Phi-3.5-mini-instruct-causal_lm-defaults_one-year-of-tsla-on-reddit-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-random-778_aps-False-mx-512_lora-16-32-o_proj_qkv_proj-0.01-True_5e-05-linear-0.01-5"

# ===== LUSTER models ===== #
# https://arxiv.org/abs/2507.01594
#
# Notes:
# - These models are fine-tuned versions of the "microsoft/Phi-3.5-mini-instruct" model,
#   but they are usually supposed to be used with a certain contrained decoding strategy.
#   Nevertheless, for testing. we plug them into our inference setup here.

# LANGUAGE_MODEL_LIST="luster-rl-sent"

# ===== Gemma models ===== #
# https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d

# LANGUAGE_MODEL_LIST="gemma-3-1b-pt" # <-- (Safetensors: 1,000M parameters)
# LANGUAGE_MODEL_LIST="gemma-3-1b-it" # <-- (Safetensors: 1,000M parameters)


# ===== LLama models ===== #

# Notes:
# - There is no 8B variant of the Llama-3.2 models, this only exists for the Llama-3.1 models.

# LANGUAGE_MODEL_LIST="Llama-3.2-1B" # <-- (Safetensors: 1.24B parameters)
# LANGUAGE_MODEL_LIST="Llama-3.2-3B" # <-- (Safetensors: 3.21B parameters)
LANGUAGE_MODEL_LIST="Llama-3.2-1B,Llama-3.2-3B"

# LANGUAGE_MODEL_LIST="Llama-3.1-8B" # <-- (Safetensors: 8.03B parameters)


# ======================== #

LANGUAGE_MODEL_ARGS=(
    "language_model=$LANGUAGE_MODEL_LIST"
    # "++language_model.checkpoint_no=400"
    # "++language_model.checkpoint_no=400,800,1200,1600,2000,2400,2800"
)

PREFERRED_TORCH_BACKEND="auto"
# PREFERRED_TORCH_BACKEND="cpu"

ADDITIONAL_OVERRIDES=""

# >>> END: Select parameters
# ==================================================== #

# NOTE:
# With torch backend set to "mps" on MacBook M1,
# the inference of the "gpt2-large" model appears to be broken.
# This is why we can set the preferred_torch_backend to "cpu" here.


COMMON_ARGS=(
    # "global_seed=1111"
    "global_seed=1111,1112"
    "preferred_torch_backend=$PREFERRED_TORCH_BACKEND"
    $ADDITIONAL_OVERRIDES
)

ARGS=(
    "${BASE_ARGS[@]}"
    "${HYDRA_LAUNCHER_ARGS[@]}"
    "${LANGUAGE_MODEL_ARGS[@]}"
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
    echo "💡 [DRY RUN] Would run:"
    echo "uv run python3 $RELATIVE_PYTHON_SCRIPT_PATH ${ARGS[*]}"
else
    uv run python3 $RELATIVE_PYTHON_SCRIPT_PATH "${ARGS[@]}"
fi

echo ">>> Calling script from RELATIVE_PYTHON_SCRIPT_PATH=${RELATIVE_PYTHON_SCRIPT_PATH} DONE"


# Exit with last command's exit code
echo ">>> Exiting with code $?."
exit $?
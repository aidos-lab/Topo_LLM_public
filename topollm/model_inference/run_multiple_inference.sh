#!/bin/bash

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
# Select the parameters here

# LANGUAGE_MODEL_LIST="roberta-base"
# LANGUAGE_MODEL_LIST="bert-base-uncased,gpt2-large,roberta-base"
# LANGUAGE_MODEL_LIST="roberta-base,roberta-base_finetuned-on-multiwoz21_ftm-lora"
# LANGUAGE_MODEL_LIST="bert-base-uncased,bert-base-uncased_finetuned-on-xsum_ftm-standard,bert-base-uncased_finetuned-on-multiwoz21_ftm-standard,bert-base-uncased_finetuned-on-sgd_ftm-lora"
# LANGUAGE_MODEL_LIST="roberta-base_finetuned-on-one-year-of-tsla-on-reddit_ftm-standard_freeze-first-6-layers_overfitted"
# LANGUAGE_MODEL_LIST="roberta-base_finetuned-on-multiwoz21_ftm-standard_full-dataset"

# LANGUAGE_MODEL_LIST="Phi-3.5-mini-instruct"
LANGUAGE_MODEL_LIST="Phi-3.5-mini-instruct-causal_lm-defaults_multiwoz21-rm-empty-True-do_nothing-ner_tags_train-10000-random-778_aps-False-mx-512_lora-16-32-o_proj_qkv_proj-0.01-True_5e-05-linear-0.01-freeze--5"
# LANGUAGE_MODEL_LIST="Phi-3.5-mini-instruct-causal_lm-defaults_one-year-of-tsla-on-reddit-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-random-778_aps-False-mx-512_lora-16-32-o_proj_qkv_proj-0.01-True_5e-05-linear-0.01-freeze--5"

LANGUAGE_MODEL_ARGS=(
    "language_model=$LANGUAGE_MODEL_LIST"
    # "++language_model.checkpoint_no=400"
)

ADDITIONAL_OVERRIDES=""

# ==================================================== #

# NOTE:
# With torch backend set to "mps" on MacBook M1,
# the inference of the "gpt2-large" model appears to be broken.
# This is why we can set the preferred_torch_backend to "cpu" here.

PREFERRED_TORCH_BACKEND="auto"
# PREFERRED_TORCH_BACKEND="cpu"

CUDA_VISIBLE_DEVICES=0

# ==================================================== #

echo "Calling python script."

uv run python3 "$RELATIVE_PYTHON_SCRIPT_PATH" \
    --multirun \
    hydra/sweeper=basic \
    hydra/launcher=basic \
    "${LANGUAGE_MODEL_ARGS[@]}" \
    preferred_torch_backend=$PREFERRED_TORCH_BACKEND \
    ++hydra.job.env_set.CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    $ADDITIONAL_OVERRIDES

# Exit with status code 0
echo "Finished running the script."
exit 0
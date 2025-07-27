#!/bin/bash

# https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/

echo "TOPO_LLM_REPOSITORY_BASE_PATH=${TOPO_LLM_REPOSITORY_BASE_PATH}"

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

LANGUAGE_MODEL_LIST="Phi-3.5-mini-instruct"

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

uv run python3 "$ABSOLUTE_PYTHON_SCRIPT_PATH" \
    --multirun \
    language_model=$LANGUAGE_MODEL_LIST \
    preferred_torch_backend=$PREFERRED_TORCH_BACKEND \
    ++hydra.job.env_set.CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    $ADDITIONAL_OVERRIDES

# Exit with status code 0
echo "Finished running the script."
exit 0
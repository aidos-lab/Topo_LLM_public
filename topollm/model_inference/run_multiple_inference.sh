#!/bin/bash

# https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/

PYTHON_SCRIPT_NAME="run_inference_pipeline.py"

# ==================================================== #
# Select the parameters here

# LANGUAGE_MODEL_LIST="bert-base-uncased,gpt2-large,roberta-base"
LANGUAGE_MODEL_LIST="roberta-base,roberta-base_finetuned-on-multiwoz21_ftm-lora"

ADDITIONAL_OVERRIDES=""
# ADDITIONAL_OVERRIDES="finetuning.max_steps=10"

# ==================================================== #

# NOTE:
# With torch backend set to "mps" on MacBook M1,
# the inference of the "gpt2-large" model appears to be broken.
# This is why we set the preferred_torch_backend to "cpu" here.

# PREFERRED_TORCH_BACKEND="auto"
PREFERRED_TORCH_BACKEND="cpu"

python3 $PYTHON_SCRIPT_NAME \
    --multirun \
    language_model=$LANGUAGE_MODEL_LIST \
    preferred_torch_backend=$PREFERRED_TORCH_BACKEND \
    $ADDITIONAL_OVERRIDES

# Exit with status code 0
exit 0
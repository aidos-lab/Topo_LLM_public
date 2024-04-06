#!/bin/bash

# https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/

PYTHON_SCRIPT_NAME="run_inference_pipeline.py"

LANGUAGE_MODEL_LIST="bert-base-uncased,gpt2-large,roberta-base"

ADDITIONAL_OVERRIDES=""
# ADDITIONAL_OVERRIDES="finetuning.max_steps=10"

# NOTE:
# With torch backend set to "mps" on MacBook M1,
# the inference of the "gpt2-large" model appears to be broken.
# This is why we set the preferred_torch_backend to "cpu" here.

python3 $PYTHON_SCRIPT_NAME \
    --multirun \
    language_model=$LANGUAGE_MODEL_LIST \
    preferred_torch_backend=cpu \
    $ADDITIONAL_OVERRIDES

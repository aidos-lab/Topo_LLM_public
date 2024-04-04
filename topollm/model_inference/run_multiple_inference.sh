#!/bin/bash

# https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/

PYTHON_SCRIPT_NAME="run_inference_pipeline.py"

LANGUAGE_MODEL_LIST="gpt2-large"

ADDITIONAL_OVERRIDES=""
# ADDITIONAL_OVERRIDES="finetuning.max_steps=10"

# NOTE:
# With torch backend set to "mps" on MacBook M1,
# the inference of the "gpt2-large" model appears to be broken.

python3 $PYTHON_SCRIPT_NAME \
    --multirun \
    embeddings/language_model=$LANGUAGE_MODEL_LIST \ # TODO: This override is not working
    preferred_torch_backend=cpu \
    $ADDITIONAL_OVERRIDES
